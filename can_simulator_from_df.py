#!/usr/bin/env python3
"""
Read a CSV into a pandas DataFrame and send CAN frames using python-can.


Example:
  python3 can_simulator_from_df.py --file normal_data.csv --bustype virtual

Defaults assume columns named feature1..feature5 and label.
"""

import argparse
import time
import logging
import sys
from typing import List

import pandas as pd
import can

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def feature_row_to_data(row, feature_cols: List[str], label_col: str):
    # feature1 scaling/truncation as in user snippet
    f1 = int(row[feature_cols[0]] * 100) & 0xFF
    f2 = int(row[feature_cols[1]]) & 0xFF
    f3 = int(row[feature_cols[2]]) & 0xFF
    f4 = int(row[feature_cols[3]]) & 0xFF

    # feature5 -> first character ordinal
    v5 = str(row[feature_cols[4]])
    c5 = ord(v5[0]) & 0xFF if v5 else 0

    label_flag = 1 if str(row[label_col]).lower() == 'normal' else 0

    return [f1, f2, f3, f4, c5, label_flag]


def main():
    p = argparse.ArgumentParser(description='Send CAN frames from a CSV via pandas DataFrame')
    p.add_argument('--file', '-f', default='normal_data.csv', help='CSV file to read')
    p.add_argument('--bustype', '-b', default='socketcan', help='python-can bustype (socketcan, virtual, etc)')
    p.add_argument('--channel', '-c', default='vcan0', help='CAN channel (vcan0)')
    p.add_argument('--period', '-p', type=float, default=0.01, help='Seconds between frames')
    p.add_argument('--id-base', type=lambda x: int(x,0), default=0x200, help='Base CAN ID (hex ok)')
    p.add_argument('--feature-cols', default='feature1,feature2,feature3,feature4,feature5', help='Comma-separated feature column names')
    p.add_argument('--label-col', default='label', help='Label column name')
    p.add_argument('--loop', action='store_true', help='Loop dataset repeatedly')
    args = p.parse_args()

    try:
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        logger.error(f'CSV file not found: {args.file}')
        sys.exit(2)

    feature_cols = [c.strip() for c in args.feature_cols.split(',')]
    if len(feature_cols) != 5:
        logger.error('feature-cols must contain 5 column names')
        sys.exit(2)

    logger.info(f'Read {len(df)} rows from {args.file}; feature columns: {feature_cols}; label: {args.label_col}')

    # open bus
    logger.info(f'Opening CAN bus bustype={args.bustype} channel={args.channel}')
    try:
        bus = can.interface.Bus(channel=args.channel, bustype=args.bustype)
    except Exception as e:
        logger.error(f'Failed to open bus: {e}\nIf on macOS use --bustype virtual for local testing')
        sys.exit(3)

    try:
        while True:
            for idx, row in df.iterrows():
                data = feature_row_to_data(row, feature_cols, args.label_col)
                arb_id = args.id_base + int(idx)
                msg = can.Message(arbitration_id=arb_id, data=bytearray(data), is_extended_id=False)
                try:
                    bus.send(msg)
                    logger.info(f'Sent frame {idx}: id=0x{arb_id:X} data={[hex(b) for b in data]}')
                except can.CanError as e:
                    logger.error(f'Failed to send message {idx}: {e}')

                time.sleep(args.period)
            if not args.loop:
                break
    except KeyboardInterrupt:
        logger.info('Interrupted by user; exiting')


if __name__ == '__main__':
    main()
