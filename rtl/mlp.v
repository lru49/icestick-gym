// iCEstick policy engine: a fixed-point 2-layer MLP (or a single linear layer)
// that maps an observation vector to K outputs + their argmax.
// host/mlp_model.py is the bit-exact golden model.
//
// Arithmetic (all signed 16-bit, ONE = 256 is appended to every layer input):
//   layer(in, W, s, relu):  o_j = sat16((sum_i W[j][i] * in[i]) >>> s), relu -> max(o_j, 0)
//   H > 0:  h = layer(x ++ [ONE], W1, S1, relu);  o = layer(h ++ [ONE], W2, S2, linear)
//   H == 0: o = layer(x ++ [ONE], W2, S2, linear)       (linear state feedback)
// Weight memory (512 x 16) holds the rows back to back: row j of layer 1 is
// W1[j][0..N-1], b1[j]; then layer 2's rows W2[k][0..], b2[k].
//
// UART (1 Mbaud 8N1, big-endian):
//   'w' ah al vh vl   addr < 0x200: weight word; 0x200 N, 0x201 H, 0x202 K,
//                     0x203 S1, 0x204 S2                          -> 'k'
//   'r' ah al         weight word; 0x2FF reads the ID 0x4D4C       -> vh vl
//   'x' x0h x0l ...   N observations                               -> argmax, o0h o0l, ...
// 'x' commands may be pipelined: up to 4 can be outstanding (one input bank
// each). A partial command is dropped after ~4 ms without bytes.
module top #(
    parameter BAUD_DIV = 12,      // 12 MHz / 12 = 1 Mbaud
    parameter MS_DIV   = 12000    // 1 ms timebase for the parser timeout
) (
    input  wire clk,              // 12 MHz, pin 21
    input  wire uart_rx,          // pin 9
    output wire uart_tx,          // pin 8
    output wire led_g,            // D5: design alive
    output wire [3:0] led_r       // D1-D4: bits of the inference count
);
    // ---------------- Config -------------------------------------------------
    reg [3:0] cfg_n = 4;          // inputs (1..14)
    reg [5:0] cfg_h = 0;          // hidden units (0..32; 0 = linear policy)
    reg [2:0] cfg_k = 1;          // outputs (1..4)
    reg [3:0] cfg_s1 = 8, cfg_s2 = 8;

    // ---------------- UART ---------------------------------------------------
    wire       rx_valid;
    wire [7:0] rx_data;
    reg        tx_start = 0;
    reg  [7:0] tx_data = 0;
    wire       tx_busy;
    uart_rx #(.DIV(BAUD_DIV)) rx0 (.clk(clk), .rx(uart_rx), .valid(rx_valid), .data(rx_data));
    uart_tx #(.DIV(BAUD_DIV)) tx0 (.clk(clk), .start(tx_start), .data(tx_data), .tx(uart_tx), .busy(tx_busy));

    reg [13:0] ms_cnt = 0;
    wire       ms = (ms_cnt == MS_DIV - 1);
    always @(posedge clk) ms_cnt <= ms ? 14'd0 : ms_cnt + 14'd1;

    // ---------------- Memories -------------------------------------------------
    reg [15:0] wmem [0:511];
    reg  [8:0] w_raddr = 0;
    reg [15:0] w_rdata = 0;
    reg        w_we = 0;
    reg  [8:0] w_waddr = 0;
    reg [15:0] w_wdata = 0;
    always @(posedge clk) begin
        if (w_we) wmem[w_waddr] <= w_wdata;
        w_rdata <= wmem[w_raddr];
    end

    // Activations: 4 banks of 64 words: x at 0..15, h at 16..48
    reg [15:0] amem [0:255];
    reg  [7:0] a_raddr = 0;
    reg [15:0] a_rdata = 0;
    wire       a_we;
    wire [7:0] a_waddr;
    wire [15:0] a_wdata;
    always @(posedge clk) begin
        if (a_we) amem[a_waddr] <= a_wdata;
        a_rdata <= amem[a_raddr];
    end

    // Parser -> amem write (priority) and FSM -> amem write
    reg        p_we = 0;
    reg  [7:0] p_waddr = 0;
    reg [15:0] p_wdata = 0;
    reg        f_we;
    reg  [7:0] f_waddr;
    reg [15:0] f_wdata;
    assign a_we    = p_we || f_we;
    assign a_waddr = p_we ? p_waddr : f_waddr;
    assign a_wdata = p_we ? p_wdata : f_wdata;

    // ---------------- Inference FSM --------------------------------------------
    localparam S_IDLE = 0, S_LSET = 1, S_NSTART = 2, S_ADDR = 3, S_WAIT = 4, S_LOAD = 5,
               S_MUL = 6, S_ACC = 7, S_SHIFT = 8, S_OUT = 9, S_CONST = 10, S_DONE = 11,
               S_RB1 = 12, S_RB2 = 13, S_RB3 = 14;
    reg  [3:0] st = S_IDLE;
    reg  [1:0] cbank = 0;           // bank being computed
    reg  [2:0] queued = 0;          // complete 'x' commands not yet computed
    reg        layer2 = 0;
    reg  [5:0] in_off = 0;          // layer input offset inside the bank
    reg  [5:0] n_in = 0, n_out = 0, i = 0, j = 0;
    reg  [3:0] shamt = 0, lshift = 0;
    reg  [8:0] wptr = 0;

    reg signed [36:0] acc = 0;      // 33 products of |2^30| fit in 37 bits
    reg signed [16:0] ph = 0;
    reg        [15:0] pl = 0;
    reg signed [15:0] mcand = 0;
    reg         [3:0] mbit = 0;

    reg signed [15:0] o0 = 0, o1 = 0, o2 = 0, o3 = 0, amax_v = 0;
    reg         [1:0] amax_i = 0;
    reg         [3:0] n_inf = 0;

    // TX handshake: FSM launches an inference reply; parser launches others
    reg        inf_go = 0;
    wire       tx_active;
    reg        rb_done = 0;         // readback word ready in rb_val
    reg [15:0] rb_val = 0;

    // From parser
    reg        x_done = 0;          // pulse: an 'x' command completed
    reg        rb_req = 0;          // weight readback requested (rb_addr)
    reg  [8:0] rb_addr = 0;

    // Signed x signed shift-add: the last (sign) bit of the multiplier subtracts
    wire signed [17:0] psum = pl[0] ? ((mbit == 15) ? ph - mcand : ph + mcand) : ph;
    wire signed [32:0] prod = {psum[17:0], pl[15:1]};   // = (A*B) after the 16th step
    wire               v_ovf = (acc[36:15] != {22{acc[36]}});
    wire signed [15:0] v16 = v_ovf ? {acc[36], {15{!acc[36]}}} : acc[15:0];
    wire signed [15:0] vout = (!layer2 && v16[15]) ? 16'sd0 : v16;

    always @(*) begin
        f_we = 0; f_waddr = 0; f_wdata = 0;
        case (st)
            S_IDLE:  if (queued != 0 && !(rb_req && !rb_done)) begin f_we = 1; f_waddr = {cbank, 2'd0, cfg_n}; f_wdata = 16'd256; end
            S_OUT:   if (!layer2) begin f_we = 1; f_waddr = {cbank, 6'd16 + j}; f_wdata = vout; end
            S_CONST: begin f_we = 1; f_waddr = {cbank, 6'd16 + cfg_h}; f_wdata = 16'd256; end
            default: ;
        endcase
    end

    always @(posedge clk) begin
        inf_go  <= 0;
        rb_done <= 0;
        if (x_done) queued <= queued + 1;
        case (st)
            S_IDLE: begin
                if (rb_req && !rb_done) begin
                    w_raddr <= rb_addr; st <= S_RB1;
                end else if (queued != 0 && !p_we) begin
                    queued <= queued + (x_done ? 3'd0 : -3'd1);
                    layer2 <= (cfg_h == 0);
                    wptr   <= 0;
                    st     <= S_LSET;
                end
            end
            S_RB1: st <= S_RB2;
            S_RB2: st <= S_RB3;
            S_RB3: begin rb_val <= w_rdata; rb_done <= 1; st <= S_IDLE; end
            S_LSET: begin
                if (!layer2) begin
                    in_off <= 0; n_in <= cfg_n; n_out <= cfg_h; lshift <= cfg_s1;
                end else begin
                    in_off  <= (cfg_h == 0) ? 6'd0 : 6'd16;
                    n_in    <= (cfg_h == 0) ? cfg_n : cfg_h;
                    n_out   <= cfg_k; lshift <= cfg_s2;
                end
                j  <= 0;
                st <= S_NSTART;
            end
            S_NSTART: begin acc <= 0; i <= 0; shamt <= lshift; st <= S_ADDR; end
            S_ADDR: begin w_raddr <= wptr; a_raddr <= {cbank, in_off + i}; st <= S_WAIT; end
            S_WAIT: st <= S_LOAD;
            S_LOAD: begin
                ph <= 0; pl <= w_rdata; mcand <= a_rdata; mbit <= 0;
                st <= S_MUL;
            end
            S_MUL: begin
                ph   <= psum >>> 1;
                pl   <= {psum[0], pl[15:1]};
                mbit <= mbit + 1;
                if (mbit == 15) begin
                    // last step: take the product straight from psum
                    acc  <= acc + prod;
                    wptr <= wptr + 1;
                    st   <= S_ACC;
                end
            end
            S_ACC: begin
                if (i == n_in) st <= S_SHIFT;
                else begin i <= i + 1; st <= S_ADDR; end
            end
            S_SHIFT: if (shamt == 0) st <= S_OUT;
                     else begin acc <= acc >>> 1; shamt <= shamt - 1; end
            S_OUT: if (!layer2) begin
                       if (!p_we) begin       // parser owns the write port this cycle
                           j <= j + 1;
                           st <= (j + 1 == n_out) ? S_CONST : S_NSTART;
                       end
                   end else if (!tx_active) begin
                       case (j[1:0])
                           0: o0 <= vout;
                           1: o1 <= vout;
                           2: o2 <= vout;
                           default: o3 <= vout;
                       endcase
                       if (j == 0 || vout > amax_v) begin amax_v <= vout; amax_i <= j[1:0]; end
                       j  <= j + 1;
                       st <= (j + 1 == n_out) ? S_DONE : S_NSTART;
                   end
            S_CONST: if (!p_we) begin layer2 <= 1; st <= S_LSET; end
            S_DONE: begin
                inf_go <= 1;
                cbank  <= cbank + 1;
                n_inf  <= n_inf + 1;
                st     <= S_IDLE;
            end
            default: st <= S_IDLE;
        endcase
    end

    // ---------------- Parser ------------------------------------------------------
    localparam C_NONE = 0, C_W = 1, C_R = 2, C_X = 3;
    reg  [1:0] cmd = C_NONE;
    reg  [4:0] nb = 0;              // argument bytes received
    reg  [7:0] b0 = 0, b1 = 0, b2 = 0;
    reg  [1:0] wbank = 0;
    reg  [2:0] stale = 0;
    reg        rep_go = 0;          // send rep_val (rep_len bytes)
    reg  [1:0] rep_len = 0;
    reg [15:0] rep_val = 0;

    wire [4:0] need = (cmd == C_W) ? 5'd4 : (cmd == C_R) ? 5'd2 : {cfg_n, 1'b0};
    wire [9:0] addr_full = {b0[1:0], b1};

    always @(posedge clk) begin
        x_done <= 0;
        p_we   <= 0;
        w_we   <= 0;
        rep_go <= 0;
        if (rb_done) begin rep_val <= rb_val; rep_len <= 2; rep_go <= 1; rb_req <= 0; end

        if (rx_valid) begin
            stale <= 0;
            if (cmd == C_NONE) begin
                nb  <= 0;
                cmd <= (rx_data == "w") ? C_W : (rx_data == "r") ? C_R :
                       (rx_data == "x") ? C_X : C_NONE;
            end else begin
                nb <= nb + 1;
                case (nb)
                    0: b0 <= rx_data;
                    1: b1 <= rx_data;
                    2: b2 <= rx_data;
                    default: ;
                endcase
                if (cmd == C_X && nb[0]) begin     // a complete observation word
                    p_we <= 1; p_waddr <= {wbank, 2'd0, nb[4:1]}; p_wdata <= {b0, rx_data};
                end
                if (cmd == C_X && !nb[0]) b0 <= rx_data;
                if (nb + 1 == need) begin
                    cmd <= C_NONE;
                    case (cmd)
                        C_W: begin
                            if (!b0[1]) begin
                                w_we <= 1; w_waddr <= {b0[0], b1}; w_wdata <= {b2, rx_data};
                            end else case (b1[2:0])
                                0: cfg_n  <= rx_data[3:0];
                                1: cfg_h  <= rx_data[5:0];
                                2: cfg_k  <= rx_data[2:0];
                                3: cfg_s1 <= rx_data[3:0];
                                4: cfg_s2 <= rx_data[3:0];
                                default: ;
                            endcase
                            rep_val <= "k"; rep_len <= 1; rep_go <= 1;
                        end
                        C_R: begin
                            if (!b0[1]) begin rb_req <= 1; rb_addr <= {b0[0], rx_data}; end
                            else begin rep_len <= 2; rep_go <= 1; rep_val <= (rx_data == 8'hff) ? 16'h4d4c : 16'd0; end
                        end
                        default: begin x_done <= 1; wbank <= wbank + 1; end
                    endcase
                end
            end
        end else if (ms && cmd != C_NONE) begin
            stale <= stale + 1;
            if (stale == 3) cmd <= C_NONE;
        end
    end

    // ---------------- TX sender ----------------------------------------------------
    reg        tinf = 0;            // current message is an inference reply
    reg  [3:0] tleft = 0, tidx = 0;
    reg        inf_pend = 0, rep_pend = 0;
    assign tx_active = (tleft != 0) || inf_pend;

    reg [7:0] tbyte;
    always @(*) begin
        if (!tinf) tbyte = (tleft == 2) ? rep_val[15:8] : rep_val[7:0];
        else case (tidx)
            0: tbyte = {6'd0, amax_i};
            1: tbyte = o0[15:8];  2: tbyte = o0[7:0];
            3: tbyte = o1[15:8];  4: tbyte = o1[7:0];
            5: tbyte = o2[15:8];  6: tbyte = o2[7:0];
            7: tbyte = o3[15:8];  default: tbyte = o3[7:0];
        endcase
    end

    always @(posedge clk) begin
        tx_start <= 0;
        if (inf_go) inf_pend <= 1;
        if (rep_go) rep_pend <= 1;
        if (tleft == 0) begin
            if (inf_pend && !inf_go) begin
                tinf <= 1; tleft <= {cfg_k, 1'b0} + 1; tidx <= 0; inf_pend <= 0;
            end else if (rep_pend && !rep_go) begin
                tinf <= 0; tleft <= {2'd0, rep_len}; rep_pend <= 0;
            end
        end else if (!tx_busy && !tx_start) begin
            tx_data  <= tbyte;
            tleft    <= tleft - 1;
            tidx     <= tidx + 1;
            tx_start <= 1;
        end
    end

    assign led_g = 1;
    assign led_r = n_inf;
endmodule
