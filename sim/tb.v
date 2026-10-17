// Byte-stream testbench: sim/stim.txt lines are "B <byte>" (send), "W <n>"
// (wait until n reply bytes have arrived) or "D <cycles>". Every byte the
// DUT sends goes to rx.txt; sim/run_test.py checks it against the model.
module tb;
    localparam BAUD_DIV = 12, MS_DIV = 2000;
    reg clk = 0;
    always #1 clk = ~clk;
    reg rx = 1;
    wire tx, led_g;
    wire [3:0] led_r;
    top #(.BAUD_DIV(BAUD_DIV), .MS_DIV(MS_DIV)) dut (
        .clk(clk), .uart_rx(rx), .uart_tx(tx), .led_g(led_g), .led_r(led_r));

    integer stim, rxlog, r, v, i, nrx = 0, t;
    reg [8*2-1:0] op;
    reg [7:0] rbyte;

    always begin
        @(negedge tx);
        repeat (BAUD_DIV / 2) @(posedge clk);
        for (i = 0; i < 8; i = i + 1) begin repeat (BAUD_DIV) @(posedge clk); rbyte[i] = tx; end
        repeat (BAUD_DIV) @(posedge clk);
        if (!tx) begin $display("FAIL: framing error"); $finish; end
        $fdisplay(rxlog, "%0d", rbyte);
        nrx = nrx + 1;
    end

    task send(input [7:0] b);
        integer k;
        begin
            rx = 0; repeat (BAUD_DIV) @(posedge clk);
            for (k = 0; k < 8; k = k + 1) begin rx = b[k]; repeat (BAUD_DIV) @(posedge clk); end
            rx = 1; repeat (BAUD_DIV) @(posedge clk);
        end
    endtask

    initial begin
        rxlog = $fopen("rx.txt", "w");
        stim  = $fopen("stim.txt", "r");
        repeat (20) @(posedge clk);
        while (!$feof(stim)) begin
            r = $fscanf(stim, "%s %d", op, v);
            if (r == 2) begin
                if (op == "B") send(v[7:0]);
                else if (op == "D") repeat (v) @(posedge clk);
                else if (op == "W") begin
                    t = 0;
                    while (nrx < v) begin
                        @(posedge clk); t = t + 1;
                        if (t > 2000000) begin $display("FAIL: timeout waiting for %0d bytes (have %0d)", v, nrx); $finish; end
                    end
                end
            end
        end
        repeat (2000) @(posedge clk);
        $fclose(rxlog);
        $display("DONE");
        $finish;
    end
endmodule
