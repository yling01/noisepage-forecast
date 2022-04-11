import os
import atexit

from constants import FORECAST_FIFO

def main():
    os.mkfifo(FORECAST_FIFO)
    atexit.register(lambda: os.remove(FORECAST_FIFO))

    while True:
        with open(FORECAST_FIFO, "r") as fifo:
            for line in fifo:
                print(line)

if __name__ == "__main__":
    main()
