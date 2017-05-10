import cnn
import numpy as np
import gradient_check as gc

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
        
        
runit = runner()
runit.lossSanityCheck()
runit.gradCheck()