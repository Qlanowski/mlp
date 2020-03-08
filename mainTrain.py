import getopt
import sys

from train.trainConfig import TrainConfig


def main(argv):
    help_txt = 'mainTrain.py ' \
               '-l <"1,2,1"> ' \
               '-af <"0" - ReLU | "1" -... |> ' \
               '-b <bias> ' \
               '-bs <batch_size> ' \
               '-ni <number_of_iterations> ' \
               '-lr <learning_rate> ' \
               '-m <momentum> -p <0 - cls | 1 - reg> ' \
               '-i <input_file>'
    try:
        opts, args = getopt.getopt(argv, "hl:f:b:s:n:r:m:p:i",
                                   ["help", "layers=", "activation_function=", "bias=", "batch_size=",
                                    "number_of_iterations=", "learning_rate=", "momentum=", "problem=", "input="])
    except getopt.GetoptError:
        print(help_txt)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_txt)
            sys.exit()
        elif opt in ("-l", "--layers"):
            layers = [int(x) for x in arg.split(",")]
        elif opt in ("-f", "--activation_function"):
            activation_function = int(arg)
        elif opt in ("-b", "--bias"):
            bias = int(arg)
        elif opt in ("-s", "--batch_size"):
            batch_size = int(arg)
        elif opt in ("-n", "--number_of_iterations"):
            number_of_iterations = int(arg)
        elif opt in ("-r", "--learning_rate"):
            learning_rate = float(arg)
        elif opt in ("-m", "--momentum"):
            momentum = float(arg)
        elif opt in ("-p", "--problem"):
            problem = int(arg)
        elif opt in ("-i", "--input"):
            input_file = arg

    return TrainConfig(layers, activation_function, bias, batch_size, number_of_iterations, learning_rate, momentum,
                       problem, input_file)


if __name__ == "__main__":
    # read config
    config = main(sys.argv[1:])
    # load data

    # initialize trainer
    # train
    # save net
