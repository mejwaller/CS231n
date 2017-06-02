import cnn
import numpy as np
import gradient_check as gc
from cifar10_preproc import PreProcCifar10
import solver as s
import matplotlib.pyplot as plt
from vis_utils import visualize_grid

class runner:
    def lossSanityCheck(self):
        model = cnn.ThreeLayerConvNet()
        N=50
        X=np.random.randn(N,3,32,32)
        y = np.random.randint(10, size=N)
        
        print "Loss, no reg. Should be about ", np.log(10)
        loss, grads = model.loss(X,y)
        print 'Initial loss (no regularization): ', loss
        
        print "Loss, with reg. Should be higher with reg,"
        model.reg = 0.5
        loss, grads = model.loss(X,y)
        print 'Initial loss (with regularization): ' , loss
        
    def gradCheck(self):
        print 'Running grad check...'
        num_inputs = 2
        input_dim = (3,16,16)
        #reg = 0.0
        num_classes = 10
        X = np.random.randn(num_inputs, *input_dim)
        y = np.random.randint(num_classes, size=num_inputs)
        
        model = cnn.ThreeLayerConvNet(num_filters=3, filter_size=3, 
                                    input_dim=input_dim, hidden_dim=7,
                                    dtype=np.float64)
        loss, grads = model.loss(X,y)
        
        for param_name in sorted(grads):
            f = lambda _: model.loss(X,y)[0]
            param_grad_num = gc.eval_numerical_gradient(f, model.params[param_name], verbose = False, h = 1e-16)
            print 'param_name: ',param_name
            print 'param_grad_num: ', param_grad_num
            print 'grads[aram_name]: ', grads[param_name]
            e = gc.rel_error(param_grad_num, grads[param_name])
            #print '%s max relative error: %e' % (param_name, gc.rel_error(param_grad_num, grads[param_name]))      
            print '%s max relative error: %e' % (param_name, e)   
            
    def smallFit(self):
        pp=PreProcCifar10()
        pp.preProcess() 
        
        print 'pp.X shape: ', pp.X_train.shape
        num_train = 100
        small_data =  {
            'X_train': pp.X_train[:num_train],
            'y_train': pp.y_train[:num_train],
            'X_val': pp.X_val,
            'y_val':pp.y_val
        }
        
        model = cnn.ThreeLayerConvNet(weight_scale = 1e-2)
        solver = s.Solver(model, small_data, num_epochs=10, batch_size=50,
                            update_rule= 'adam',
                            optim_config={
                                'learning_rate': 1e-3,
                            },
                            verbose = True, print_every = 1)
        print 'Running fit of small data...'
        solver.train()
        
        print 'Solving finished, plotting...'
        plt.subplot(2, 1, 1)
        plt.plot(solver.loss_history, 'o')
        plt.xlabel('iteration')
        plt.ylabel('loss')

        plt.subplot(2, 1, 2)
        plt.plot(solver.train_acc_history, '-o')
        plt.plot(solver.val_acc_history, '-o')
        plt.legend(['train', 'val'], loc='upper left')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.show()

    def oneEpoch(self):
        pp=PreProcCifar10()
        pp.preProcess()

        data = {
            'X_train': pp.X_train,
            'y_train': pp.y_train,
            'X_val': pp.X_val,
            'y_val': pp.y_val
        }

        model = cnn.ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg = 0.001)

        solver = s.Solver(model, data,
                        num_epochs=1, batch_size=50, update_rule='adam',
                        optim_config={
                            'learning_rate': 1e-3,
                        }, verbose=True, print_every=20)
        solver.train()

        print 'Solving finished, plotting...'
        plt.subplot(2, 1, 1)
        plt.plot(solver.loss_history, 'o')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.subplot(2, 1, 2)
        plt.plot(solver.train_acc_history, '-o')
        plt.plot(solver.val_acc_history, '-o')
        plt.legend(['train', 'val'], loc='upper left')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.show()


        grid = visualize_grid(model.params['W1'].transpose(0,2,3,1))
        plt.imshow(grid.astype('uint8'))
        plt.axis('off')
        plt.gcf().set_size_inches(5,5)
        plt.show()
        
        
runit = runner()
runit.lossSanityCheck()
runit.gradCheck()
runit.smallFit()
runit.oneEpoch()
