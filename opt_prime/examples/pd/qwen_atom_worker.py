"""Compatibility entry point; the worker implementation lives in the library."""

from opt_prime.accel.pd.npu.worker import main

if __name__ == "__main__":
    main()
