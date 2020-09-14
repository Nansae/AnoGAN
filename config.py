
class Config(object):
    def display(self):
        print("\nConfigurations:")
        for i in dir(self):
            if not i.startswith('__') and not callable(getattr(self, i)):
                print(' {:30} {}'.format(i, getattr(self, i)))
        print('\n')

    MODE = ["segment", "decision", "total"]

    #PATH
    PATH_DATA = "data"
    PATH_CHECKPOINT = "checkpoints"
    PATH_ANOGAN_CHECKPOINT = "checkpoints_anogan"
    PATH_TENSORBOARD = "tensorboard"

    #PARA
    EPOCH = 50
    BATCH_SIZE = 10
    MAX_TO_KEEP = 10
    #LEARNING_RATE = 0.00001
    #LEARNING_RATE = 0.0001

    ##
    IMG_SIZE = 256
    LATENT_DIM = 100

    ##
    IS_TRAINING = True
    IS_CONTINUE = False

    ##
    PRINT_STEP = 500
    VAL_NUM = 2
    CHECKPOINTS_STEP = 1
    SUMMARY_STEP = 500

    ##
    H_FLIP = True
    V_FLIP = False
    BRIGHTNESS = 0.2
    ROTATION = None

if __name__ == "__main__":

    parameter = Config()
    parameter.display()
