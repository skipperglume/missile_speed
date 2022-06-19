import os
from  Cut_Sound import CutSound
import argparse

if __name__ == "__main__":



    print("+I+")

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i','--input', type=str,help='input file path')
    parser.add_argument('-o','--output', type=str,help='path to the output file')
    parser.add_argument('-l','--length', type=int,choices= [4000,8000,12000], default=4000,help='length of output sound file')

    args = parser.parse_args()
    # print(args.accumulate(args.integers))
    # input_name = ""
    # output_name = ""
    # sample_length = 4000
    # python3 Run_Manipulation.py --input "/mnt/c/summer_2022/input/2.wav" --output "/mnt/c/summer_2022/missile_frequency/output_sound" --length 4000
    # python3 Run_Manipulation.py --input "/mnt/c/summer_2022/input/RC.wav" --output "/mnt/c/summer_2022/missile_frequency/output_sound" --length 4000
    # print(args.input)
    # print(args.output)
    # print(args.length)
    CutSound(args.input, args.output, args.length)
