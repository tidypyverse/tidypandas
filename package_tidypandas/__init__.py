"""
This module is callable via the command line
but importing it does nothing
"""

from colored import fg, bg, attr
import pandas as pd

def main():
    print(f"{fg('dark_orange')}{attr('bold')}You called me!{attr('reset')}")


if __name__ == "__main__":
    main()
