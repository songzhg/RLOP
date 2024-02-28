import sys
import os

path = os.path.abspath(__file__)
for _ in range(4):
    path = os.path.dirname(path)
sys.path.append(path)

print(sys.path)

from python.utils import csv_to_line_chart


if __name__ == '__main__':
    csv_to_line_chart(os.path.join(path, 'data/ppo/lunar_lander/rlop_0_log.txt'))
