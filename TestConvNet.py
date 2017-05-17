import numpy as np
import matplotlib.pyplot as plt
import unittest
import layers
import gradient_check as gc
import layer_utils as lu


class ConvNetTest(unittest.TestCase):
    
    def setUp(self):
        self.x_shape = (2,3,4,4)
        self.w_shape = (3,3,4,4)
        self.x=np.linspace(-0.1,0.5,num=np.prod(self.x_shape)).reshape(self.x_shape)
        self.w=np.linspace(-0.2,0.3,num=np.prod(self.w_shape)).reshape(self.w_shape)
        self.b=np.linspace(-0.1,0.2,num=3) 
        
        self.naivefwd_correct_out = np.array([[[[[-0.08759809, -0.10987781],
                           [-0.18387192, -0.2109216 ]],
                          [[ 0.21027089,  0.21661097],
                           [ 0.22847626,  0.23004637]],
                          [[ 0.50813986,  0.54309974],
                           [ 0.64082444,  0.67101435]]],
                         [[[-0.98053589, -1.03143541],
                           [-1.19128892, -1.24695841]],
                          [[ 0.69108355,  0.66880383],
                           [ 0.59480972,  0.56776003]],
                          [[ 2.36270298,  2.36904306],
                           [ 2.38090835,  2.38247847]]]]])
                           
        self.maxpoolfwd_correct_out =  np.array([[[[-0.26315789, -0.24842105],
                          [-0.20421053, -0.18947368]],
                         [[-0.14526316, -0.13052632],
                          [-0.08631579, -0.07157895]],
                         [[-0.02736842, -0.01263158],
                          [ 0.03157895,  0.04631579]]],
                        [[[ 0.09052632,  0.10526316],
                          [ 0.14947368,  0.16421053]],
                         [[ 0.20842105,  0.22315789],
                          [ 0.26736842,  0.28210526]],
                         [[ 0.32631579,  0.34105263],
                          [ 0.38526316,  0.4       ]]]])
                           
    def testNaiveFWdPass(self):
        print 'Testing conv_forward_naive'
        conv_param = {'stride':2,'pad':1}
        out,_ = layers.conv_forward_naive(self.x,self.w,self.b,conv_param)
        #print out
        #print self.naivefwd_correct_out
        self.failUnless(np.allclose(out,self.naivefwd_correct_out))    
            
        print 'difference: ', gc.rel_error(out, self.naivefwd_correct_out)
        
    def testNaiveBackPass(self):
        print 'Testing conv_backward_naive function'
        x = np.random.randn(4,3,5,5)
        w = np.random.randn(2,3,3,3)
        b = np.random.randn(2,)
        dout = np.random.randn(4,2,5,5)
        conv_param = {'stride': 1, 'pad': 1}
        
        dx_num = gc.eval_numerical_gradient_array(lambda x: layers.conv_forward_naive(x,w,b,conv_param)[0], x, dout)
        dw_num = gc.eval_numerical_gradient_array(lambda w: layers.conv_forward_naive(x,w,b,conv_param)[0], w, dout)
        db_num = gc.eval_numerical_gradient_array(lambda b: layers.conv_forward_naive(x,w,b,conv_param)[0], b, dout)
        
        out, cache = layers.conv_forward_naive(x,w,b,conv_param)
        dx, dw, db = layers.conv_backward_naive(dout, cache)        
        
        print 'dx error: ', gc.rel_error(dx, dx_num)
        print 'dw error: ', gc.rel_error(dw, dw_num)
        print 'db error: ', gc.rel_error(db, db_num)

        
        self.failUnless(np.allclose(dx, dx_num))
        self.failUnless(np.allclose(dw, dw_num))
        self.failUnless(np.allclose(db, db_num))
        
    def testMaxPoolNaiveFwd(self):
        print 'Testing max_pool_forward_naive function:'
        x_shape=(2,3,4,4)
        x = np.linspace(-0.3,0.4,num=np.prod(x_shape)).reshape(x_shape)
        pool_param ={'pool_width':2, 'pool_height':2, 'stride':2}
        
        out,_ = layers.max_pool_forward_naive(x, pool_param)        
        
        print 'difference:', gc.rel_error(out, self.maxpoolfwd_correct_out)
        
        self.failUnless(np.allclose(out,self.maxpoolfwd_correct_out))
        
    def testMaxPoolNaiveBack(self):
        print 'Testing max_pool_backward_naive function:'
        x = np.random.randn(3,2,8,8)
        dout = np.random.randn(3,2,4,4)
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        dx_num = gc.eval_numerical_gradient_array(lambda x: layers.max_pool_forward_naive(x, pool_param)[0], x, dout)
        out, cache = layers.max_pool_forward_naive(x, pool_param)
        
        dx = layers.max_pool_backward_naive(dout, cache)        
        
        print 'dx error: ', gc.rel_error(dx, dx_num)
        
        self.failUnless(np.allclose(dx,dx_num))
        
    def testConvReluPool(self):
        print 'Testing conv_relu_pool'
        x = np.random.randn(2,3,16,16)
        w = np.random.randn(3,3,3,3)
        b = np.random.randn(3,)
        dout = np.random.randn(2,3,8,8)
        conv_param = {'stride': 1, 'pad': 1}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        out, cache = lu.conv_relu_pool_forward(x,w,b,conv_param,pool_param)
        dx, dw,db = lu.conv_relu_pool_backward(dout, cache)
        
        dx_num = gc.eval_numerical_gradient_array(lambda x: lu.conv_relu_pool_forward(x,w,b,conv_param, pool_param)[0], x, dout)
        dw_num = gc.eval_numerical_gradient_array(lambda w: lu.conv_relu_pool_forward(x,w,b,conv_param, pool_param)[0], w, dout)
        db_num = gc.eval_numerical_gradient_array(lambda b: lu.conv_relu_pool_forward(x,w,b,conv_param, pool_param)[0], b, dout)
        
        print 'dx error:', gc.rel_error(dx_num, dx)
        print 'dw error:', gc.rel_error(dw_num, dw)
        print 'db error:', gc.rel_error(db_num, db)
        
        self.failUnless(np.allclose(dx, dx_num))
        self.failUnless(np.allclose(dw, dw_num))
        self.failUnless(np.allclose(db, db_num))
        
    def testConvRelu(self):
        print 'Testing conv_relu:'
        x = np.random.randn(2,3,8,8)
        w = np.random.randn(3,3,3,3)
        b = np.random.randn(3,)
        dout = np.random.randn(2,3,8,8)
        conv_param = {'stride': 1, 'pad': 1}
        
        out, cache = lu.conv_relu_forward(x,w,b,conv_param)
        dx, dw, db = lu.conv_relu_backward(dout,cache)
        
        dx_num = gc.eval_numerical_gradient_array(lambda x: lu.conv_relu_forward(x,w,b,conv_param)[0],x,dout)
        dw_num = gc.eval_numerical_gradient_array(lambda w: lu.conv_relu_forward(x,w,b,conv_param)[0],w,dout)
        db_num = gc.eval_numerical_gradient_array(lambda b: lu.conv_relu_forward(x,w,b,conv_param)[0],b,dout)        
        
        print 'dx error:', gc.rel_error(dx_num, dx)
        print 'dw error:', gc.rel_error(dw_num, dw)
        print 'db error:', gc.rel_error(db_num, db)
        
        self.failUnless(np.allclose(dx, dx_num))
        self.failUnless(np.allclose(dw, dw_num))
        self.failUnless(np.allclose(db, db_num))
        
        
        
        
        
        
        
        
        
        
if __name__ == '__main__': unittest.main()     
    
        