// 8N1 UART, DIV = clk_hz / baud (12 MHz / 115200 ~= 104, 0.16% error)

module uart_rx #(parameter DIV = 104) (
    input  wire       clk,
    input  wire       rx,
    output reg        valid = 0,   // 1-cycle pulse when data is ready
    output reg  [7:0] data = 0
);
    reg [1:0] sync = 2'b11;        // metastability synchroniser
    always @(posedge clk) sync <= {sync[0], rx};
    wire s = sync[1];

    reg [1:0] state = 0;
    reg [7:0] cnt = 0;
    reg [2:0] bitn = 0;
    reg [7:0] sh = 0;

    always @(posedge clk) begin
        valid <= 0;
        case (state)
            0: if (!s) begin state <= 1; cnt <= DIV / 2; end          // start edge
            1: if (cnt == 0) begin                                     // mid start bit
                   if (!s) begin state <= 2; cnt <= DIV - 1; bitn <= 0; end
                   else state <= 0;                                    // glitch
               end else cnt <= cnt - 1;
            2: if (cnt == 0) begin                                     // mid data bit
                   sh <= {s, sh[7:1]};
                   cnt <= DIV - 1;
                   if (bitn == 7) state <= 3; else bitn <= bitn + 1;
               end else cnt <= cnt - 1;
            3: if (cnt == 0) begin                                     // mid stop bit
                   state <= 0;
                   if (s) begin valid <= 1; data <= sh; end            // drop framing errors
               end else cnt <= cnt - 1;
        endcase
    end
endmodule

module uart_tx #(parameter DIV = 104) (
    input  wire       clk,
    input  wire       start,       // ignored while busy
    input  wire [7:0] data,
    output reg        tx = 1,
    output wire       busy
);
    reg [9:0] sh = 10'h3ff;
    reg [3:0] n = 0;               // 11 = start + 8 data + stop + full stop-bit hold
    reg [7:0] cnt = 0;
    assign busy = (n != 0);

    always @(posedge clk) begin
        if (n == 0) begin
            if (start) begin sh <= {1'b1, data, 1'b0}; n <= 11; cnt <= 0; end
        end else if (cnt == 0) begin
            tx  <= sh[0];
            sh  <= {1'b1, sh[9:1]};
            n   <= n - 1;
            cnt <= DIV - 1;
        end else cnt <= cnt - 1;
    end
endmodule
