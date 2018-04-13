import argparse, os, time
from transfer import transfer

def parse_args():
    desc = "Transfer Learning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-e','--epoch', type=int, default=25,
                        help='Iteration of the whole dataset')
    parser.add_argument('-sz','--img_size', type=int, default=128,
                        help='The size of image')
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-b','--batch_size', type=int, default=64,
                        help='The size of batch')
    parser.add_argument('-r','--result_folder', type=str, default='results',
                        help='Folder name to save results')
    parser.add_argument('-d','--dataset', type=str, default='wood',
                        help='Dataset name')
    parser.add_argument('-p','--predict', type=bool, default=False,
                        help='Predict?')
    parser.add_argument('--decay', type=int, default=0)
    return check_args(parser.parse_args())

def check_args(args):
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    try:
        assert args.epoch >= 1
    except:
        print("Epoch number must be integer and larger than 0")

    try:
        assert args.batch_size >= 1
    except:
        print("Batch size must be integer and larger than 0")

    return args


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    # call transfer learning object
    tr = transfer(args)
    if not args.predict:
        # training
        tr.train()
        print("[*] Training finished!")
    else:
        print("[*] Start predicting...")
        # predict the generated data
        tr.load()
        tr.predict_test()
        #tr.predict_gen()
        print("[*] Prediction finished")

if __name__ == '__main__':
    main()
