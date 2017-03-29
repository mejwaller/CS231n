from cifar10_preproc import PreProcCifar10
from TwoLayerNN import TwoLayerNN
import matplotlib.pyplot as plt
from vis_utils import visualize_grid
import numpy as np

pp=PreProcCifar10()

pp.preProcess()

input_size = 32*32*3#32 x 32 images, RGB
hidden_size=50#size of hidden layer - 50 arbitrarily chosen
num_classes = 10#number of classifications possible


nn2 = TwoLayerNN(input_size, hidden_size, num_classes,1e-5)

#Train the network
stats = nn2.train(pp.X_train, pp.y_train, pp.X_val, pp.y_val,
            num_iters=10000, batch_size=300,
            learning_rate=0.0025, learning_rate_decay=0.95,
            reg= 0.1, verbose=True)
            
#predict on validation set
val_acc = (nn2.predict(pp.X_val)==pp.y_val).mean()
print 'Validation accuracy: ', val_acc

#predict on test set
test_acc = (nn2.predict(pp.X_test) == pp.y_test).mean()
print 'Test accuracy: ', test_acc


plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()


"""
best_val = -1
best_stats = None
learning_rates = [1e-4,1e-3, 1e-2]
regularization_strengths = [0.2,0.3,0.4, 0.5,]
stds=[1e-5]
results = {} 
iters = 2000 #100
for lr in learning_rates:
    for rs in regularization_strengths:
        for std in stds:
            nn2 = TwoLayerNN(input_size, hidden_size, num_classes, std)
        
            stats = nn2.train(pp.X_train, pp.y_train, pp.X_val, pp.y_val,
                num_iters=iters, batch_size=200,
                learning_rate=lr, learning_rate_decay=0.95,
                reg=rs, verbose=True)
            
        y_train_pred = nn2.predict(pp.X_train)
        acc_train = np.mean(pp.y_train == y_train_pred)
        y_val_pred = nn2.predict(pp.X_val)
        acc_val = np.mean(pp.y_val == y_val_pred)
    
        results[(lr, rs, std)] = (acc_train, acc_val)
            
        if best_val < acc_val:
            best_stats = stats
            best_val = acc_val
            best_net = nn2
            
# Print out results.
for lr, reg, std in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg, std)]
    print 'std %e lr %e reg %e train accuracy: %f val accuracy: %f' % (
                std, lr, reg, train_accuracy, val_accuracy)
    
print 'best validation accuracy achieved during cross-validation: %f' % best_val    


"""
# Visualize the weights of the network

def show_net_weights(net):
  W1 = net.params['W1']
  W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
  plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
  plt.gca().axis('off')
  plt.show()

show_net_weights(nn2)

"""
# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(best_stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(best_stats['train_acc_history'], label='train')
plt.plot(best_stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()
"""