# Bitstream + training run in GitHub Actions; locally only FPGA I/O.
SRC = rtl/mlp.v rtl/uart.v
B = build
REPO = lru49/icestick-gym
PY = .venv/bin/python
ICESTICK ?= $(shell for d in /sys/bus/usb/devices/*; do \
  [ "$$(cat $$d/idProduct 2>/dev/null)" = 6010 ] && [ "$$(cat $$d/product)" = "Dual RS232-HS" ] && \
  printf 'd:%03d/%03d' $$(cat $$d/busnum) $$(cat $$d/devnum); done)

all: $(B)/mlp.bin

$(B)/mlp.json: $(SRC)
	mkdir -p $(B)
	yosys -q -l $(B)/yosys.log -p 'read_verilog $(SRC); synth_ice40 -abc9 -top top -json $@; tee -q -o $(B)/stat.txt stat'

$(B)/mlp.asc: $(B)/mlp.json icestick.pcf
	for seed in 1 2 3 4 5 6 7 8; do \
	  nextpnr-ice40 --hx1k --package tq144 --freq 12 --pcf icestick.pcf --json $< --asc $@ --seed $$seed \
	    > $(B)/pnr.log 2>&1 && { echo "placed with seed $$seed"; break; }; \
	  grep -E 'ERROR|ICESTORM_LC' $(B)/pnr.log; rm -f $@; done; test -f $@
	grep -A8 'Device utilisation' $(B)/pnr.log
	grep 'Max frequency' $(B)/pnr.log | tail -1

$(B)/mlp.bin: $(B)/mlp.asc
	icepack $< $@

test-sim:
	$(PY) sim/run_test.py --seeds 16

# --- local: FPGA I/O only ----------------------------------------------------
fetch-bit:
	rm -rf $(B)/ci && gh run download -R $(REPO) -n bitstream -D $(B)/ci && cat $(B)/ci/summary.txt

flash:
	@test -n "$(ICESTICK)" || { echo "iCEstick not found"; exit 1; }
	iceprog -d $(ICESTICK) $(B)/ci/mlp.bin

test-hw:
	sg dialout -c '$(PY) host/hwtest.py'

# RUN=<id of a successful train workflow run>
fetch-policies:
	rm -rf $(B)/policies && mkdir -p $(B)/policies && \
	gh run download -R $(REPO) $(RUN) -p 'policy-*' -D $(B)/policies && ls $(B)/policies

replay:
	sg dialout -c '$(PY) host/replay.py $(B)/policies/*/'

.PHONY: all test-sim fetch-bit flash test-hw fetch-policies replay
