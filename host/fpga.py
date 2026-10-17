"""Host driver for the iCEstick policy engine (rtl/mlp.v)."""
import time

import numpy as np
import serial

import mlp_model as M

PORT = "/dev/serial/by-id/usb-FTDI_Dual_RS232-HS-if01-port0"


def word(v):
    return bytes([(v >> 8) & 255, v & 255])


def s16(h, l):
    return (((h << 8) | l) ^ 0x8000) - 0x8000


class Fpga:
    def __init__(self, port=PORT, baud=1_000_000):
        self.s = serial.Serial(port, baud, timeout=2)
        try:
            self.s.set_low_latency_mode(True)     # FTDI latency timer 16 ms -> 1 ms
        except Exception:
            pass
        time.sleep(0.05)
        self.s.reset_input_buffer()
        self.K = None

    def _read(self, n):
        b = self.s.read(n)
        if len(b) != n:
            raise IOError("timeout: %d of %d bytes" % (len(b), n))
        return b

    def read(self, addr):
        self.s.write(b"r" + word(addr))
        b = self._read(2)
        return (b[0] << 8) | b[1]

    def load(self, pol, verify=True):
        """Write config + weights (pipelined), then read every word back."""
        M.check_shape(pol)
        writes = M.config_words(pol) + list(enumerate(M.pack(pol)))
        self.s.write(b"".join(b"w" + word(a) + word(v) for a, v in writes))
        if self._read(len(writes)) != b"k" * len(writes):
            raise IOError("bad write acks")
        if verify:
            image = M.pack(pol)
            self.s.write(b"".join(b"r" + word(a) for a in range(len(image))))
            b = self._read(2 * len(image))
            back = [(b[2 * i] << 8) | b[2 * i + 1] for i in range(len(image))]
            if back != image:
                raise IOError("weight readback mismatch")
            for a, v in M.config_words(pol):
                if self.read(a) != v:
                    raise IOError("config readback mismatch at 0x%x" % a)
        self.K, self.N = pol["K"], pol["N"]

    def infer(self, x):
        self.s.write(b"x" + b"".join(word(int(v)) for v in x))
        b = self._read(1 + 2 * self.K)
        return [s16(b[1 + 2 * i], b[2 + 2 * i]) for i in range(self.K)], b[0]

    def stream(self, xs, window=4):
        """Pipelined inference over many observation vectors (<= 4 in flight)."""
        n, rl = len(xs), 1 + 2 * self.K
        pkts = [b"x" + b"".join(word(int(v)) for v in x) for x in xs]
        out, sent = [], 0
        buf = b""
        while len(out) < n:
            while sent < n and sent - len(out) < window:
                self.s.write(pkts[sent]); sent += 1
            buf += self._read(rl)
            while len(buf) >= rl:
                b, buf = buf[:rl], buf[rl:]
                out.append(([s16(b[1 + 2 * i], b[2 + 2 * i]) for i in range(self.K)], b[0]))
        return out
