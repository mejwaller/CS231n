import data_utils as du
import numpy as np
import matplotlib.pyplot as plt


class PreProcCifar10:
    def __init__(self):
        self.X_train, self.y_train, self.X_test, self.y_test = du.load_CIFAR10('datasets/cifar-10-batches-py/')

        

    def sanity(self):
    
        # As a sanity check, we print out the size of the training and test data.
        print 'Training data shape: ', self.X_train.shape
        print 'Training labels shape: ', self.y_train.shape
        print 'Test data shape: ', self.X_test.shape
        print 'Test labels shape: ', self.y_test.shape

    def visualise(self):
        # Visualize some examples from the dataset.
        # We show a few examples of training images from each class.
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = len(classes)
        samples_per_class = 7
        for y, cls in enumerate(classes):
            idxs = np.flatnonzero(self.y_train == y)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(samples_per_class, num_classes, plt_idx)
                plt.imshow(self.X_train[idx].astype('uint8'))
                plt.axis('off')
                if i == 0:
                    plt.title(cls)
        plt.show()
        
    def preProcess(self):
        # Split the data into train, val, and test sets. In addition we will
        # create a small development set as a subset of the training data;
        # we can use this for development so our code runs faster.
        num_training = 49000
        num_validation = 1000
        num_test = 1000
        num_dev = 500

        # Our validation set will be num_validation points from the original
        # training set.
        mask = range(num_training, num_training + num_validation)
        self.X_val = self.X_train[mask]
        self.y_val = self.y_train[mask]

        # Our training set will be the first num_train points from the original
        # training set.
        mask = range(num_training)
        self.X_train = self.X_train[mask]
        self.y_train = self.y_train[mask]

        # We will also make a development set, which is a small subset of
        # the training set.
        mask = np.random.choice(num_training, num_dev, replace=False)
        self.X_dev = self.X_train[mask]
        self.y_dev = self.y_train[mask]
        
        # Make a fixed dev set too so that we can reproduce results for tests
        mask = [1566, 3627, 4236, 2143, 1696, 4143, 1358, 3037, 2812, 1261, 1146,
       4403, 3571, 4294, 1787, 1650, 2224,  295, 2808, 1834, 3637, 2422,
        346, 1760,  706, 4562,  700, 3969, 3181,  592, 2720, 1708,  211,
       2869, 1310, 2155, 1743, 4583, 4428, 1249, 1961, 2148,  870, 3596,
        594, 3482, 4202, 3836, 1394,  140, 1131,   76,  168, 1368, 4661,
       4821, 1342, 2187,  410, 4690, 1953, 1547, 4252, 3542, 1210,  400,
       1415,  699,  312, 3111, 1337,   31, 2462,  873, 4777,  902, 2757,
       4789, 3495, 2424,  616,  965, 2273, 1031, 4432, 3553,  866, 3390,
        926, 4891,  500, 3773, 2441, 1765, 1264, 1209, 3504,  992, 3394,
       2204, 1198, 4057, 2629, 1606, 4604, 4384, 2066, 4733,  411, 3966,
       1105, 1619, 3029, 2759,   63,   41, 4610, 2505,  184, 2535, 4240,
        691, 3117, 4859, 4166, 4697, 4410, 1958, 3919, 3719, 3921,   71,
       4569, 1319, 2133, 2580, 1922, 3823, 3401, 1403, 3064, 2513, 4638,
       4329, 1789, 4554, 2826, 1612,  672,  710,  448, 4725, 4165, 2746,
         34, 2316, 1911,  543, 1893, 1762,  183, 1638, 4836,  831, 4505,
       2425, 2057, 3535,  760, 3367, 1073, 3577, 3087, 2256, 2819, 1646,
       4248, 2312, 1309,  641, 4345, 2457, 3856, 3462, 4107,   65, 3258,
        224, 2993,  175, 2418, 2840, 1360, 4181, 2979,  551, 4075, 2853,
       1108, 2439,  878, 2004, 1107, 1325, 2260, 3060,  487, 4817, 2595,
         43, 1744, 4726, 2475, 4293,  130, 3046, 4002, 2332, 4320,  520,
       1864, 2154, 2030, 1327, 2464,  713, 1130, 2742, 3840, 1572, 3377,
       4719, 1701, 4834, 1769, 1861,  285, 1718, 3244, 3266, 4308, 1216,
       2191, 4037, 4047, 3879, 1392,  635, 2577, 4203, 4493,  744, 2889,
       3801, 4572, 2279, 2905, 1120, 3817, 1917, 1793, 1441, 3465, 2888,
       4322, 4065,  560, 1024, 3625, 2335, 4276, 4827, 1567, 1035, 1190,
       1749, 3329,  778, 2188, 3716,  546, 4445, 3873, 3470, 1737, 3552,
        552, 2786,  656,  494,  916, 3644, 4411, 1587, 4683, 1563, 3105,
       3884, 1657,  614,  531, 4348, 3044, 3413, 4340, 4829, 3486, 2511,
       3539, 4283,  904, 3572, 2106,  106,  265,   68, 1659,  381, 2800,
       1711, 3974, 3358, 1939, 3976, 4207, 1827, 3239, 1027, 1192, 4748,
       3935, 1652,  933, 2877, 2310, 3115, 2818, 3980, 3848,  562, 2861,
       4315, 4278, 1185, 2794, 2974, 3654, 4522, 3646, 3804, 1570,  549,
       1540, 2627,  623, 1086, 1229, 2212, 4170, 1047, 3955, 2798, 1206,
       4399, 3994, 2528, 3886, 1991, 2899, 4094, 4010, 2385, 2378, 3805,
       3566, 4780,  351, 3947, 4628,  135, 2696, 2548, 3189, 2811, 1675,
       1431, 1125, 3073, 4740, 1434, 1212, 2635, 3198,  107, 3613, 2437,
       1064, 1196, 3041, 2904, 4510, 4233, 4896,  869, 3597, 3085, 4257,
       2847, 4211, 2865, 4516, 1929, 1776, 2463, 1076, 3144, 4645, 1382,
       3846, 1475, 3683, 2732,  391, 2176,  394, 4297,  732, 3871,  799,
       3049, 4586,  714, 1002, 3523,  477, 2474, 3493, 1666,  707, 3250,
       4716, 4324, 3220, 2943,  318, 2863, 3954, 2175, 3437, 4379, 4005,
       1349, 4779, 2094,  245, 4517, 3061, 4159, 4332,  890,  115, 2044,
       4306, 4318, 3114, 4884, 4205, 3232,  694, 1984, 1978, 4328, 2725,
       1091, 2843,  288, 2961, 2001, 1374, 3206, 3384, 3460, 3600, 2991,
       2479, 4866,  197, 2633, 2770, 4408,  467, 2240, 3564, 3346, 3211,
       3695, 4393, 2996, 2472, 1491]
       
        self.fixedX_dev = self.X_train[mask]
        self.fixedy_dev = self.y_train[mask]

        # We use the first num_test points of the original test set as our
        # test set.
        mask = range(num_test)
        self.X_test = self.X_test[mask]
        self.y_test = self.y_test[mask]
        
        print 'Train data shape: ', self.X_train.shape
        print 'Train labels shape: ', self.y_train.shape
        print 'Validation data shape: ', self.X_val.shape
        print 'Validation labels shape: ', self.y_val.shape
        print 'Test data shape: ', self.X_test.shape
        print 'Test labels shape: ', self.y_test.shape
        
        # Preprocessing: reshape the image data into rows
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1))
        self.X_val = np.reshape(self.X_val, (self.X_val.shape[0], -1))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], -1))
        self.X_dev = np.reshape(self.X_dev, (self.X_dev.shape[0], -1))
        self.fixedX_dev = np.reshape(self.fixedX_dev, (self.fixedX_dev.shape[0], -1))
        
        # As a sanity check, print out the shapes of the data
        print 'Training data shape: ', self.X_train.shape
        print 'Validation data shape: ', self.X_val.shape
        print 'Test data shape: ', self.X_test.shape
        print 'dev data shape: ', self.X_dev.shape
        print 'fixed dev data shape: ', self.fixedX_dev.shape
        
        # Preprocessing: subtract the mean image
        # first: compute the image mean based on the training data
        mean_image = np.mean(self.X_train, axis=0)
        print mean_image[:10] # print a few of the elements
        plt.figure(figsize=(4,4))
        plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image
        plt.show()
        
        # second: subtract the mean image from train and test data
        self.X_train -= mean_image
        self.X_val -= mean_image
        self.X_test -= mean_image
        self.X_dev -= mean_image
        self.fixedX_dev -= mean_image
        
        # third: append the bias dimension of ones (i.e. bias trick) so that our SVM
        # only has to worry about optimizing a single weight matrix W.
        self.X_train = np.hstack([self.X_train, np.ones((self.X_train.shape[0], 1))])
        self.X_val = np.hstack([self.X_val, np.ones((self.X_val.shape[0], 1))])
        self.X_test = np.hstack([self.X_test, np.ones((self.X_test.shape[0], 1))])
        self.X_dev = np.hstack([self.X_dev, np.ones((self.X_dev.shape[0], 1))])
        self.fixedX_dev = np.hstack([self.fixedX_dev, np.ones((self.fixedX_dev.shape[0], 1))])


        print self.X_train.shape, self.X_val.shape, self.X_test.shape, self.X_dev.shape, self.fixedX_dev.shape
        
       
#pp = PreProcCifar10()

#pp.sanity()
#pp.visualise()