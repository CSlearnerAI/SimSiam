import argparse
import simsiam
import eval

parser = argparse.ArgumentParser(description='The framework of SimSiam')
parser.add_argument('--eval', action='store_true', default=False, help="Whether chooses to eval the model"
                                                                       "(default: False)")

if __name__ == '__main__':
    args = parser.parse_args()
    if args.eval is False:
        simsiam.main()
    else:
        eval.main()
