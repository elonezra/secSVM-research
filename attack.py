
from scipy import sparse
import numpy as np
import elon_secsvm
import pickle
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from termcolor import cprint
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import foolbox as fb



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_metrics(y_pred,labels):
    # calculate evaluation metrics with sklearn
    accuracy = accuracy_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)
    # print evaluation metrics
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1 score:", f1)


# CW-L2 Attack
# Based on the paper, i.e. not exact same version of the code on https://github.com/carlini/nn_robust_attacks
# (1) Binary search method for c, (2) Optimization on tanh space, (3) Choosing method best l2 adversaries is NOT IN THIS CODE.
def cw_l2_attack(clf, images, labels, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01) :

    images = images.toarray()  # convert the sparse matrix to a dense numpy array
    labels = torch.from_numpy(labels)
    labels = labels.to(device)
    labels = labels.reshape(-1)

    # Define f-function
    def f(x) :
        outputs = clf.model(x)
        one_hot_labels = torch.eye(len(outputs))[labels].to(device)
        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        
        # If targeted, optimize for making the other class most likely 
        if targeted :
            return torch.clamp(i-j, min=-kappa)
        
        # If untargeted, optimize for making the other class most likely 
        else :
            return torch.clamp(j-i, min=-kappa)
        


    prev = 1e10
    float_tensor_cons = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    x = float_tensor_cons(images)
    w = torch.zeros_like(x, requires_grad=True).to(device)
    # s = images.shape
    # w = torch.zeros(s)

    optimizer = optim.Adam([w], lr=learning_rate)
    output = clf.model(x).to(device)
    
    for step in range(max_iter) :
        
        a = 1/2*(torch.tanh(w) + 1)
        loss1 = nn.MSELoss(reduction='sum')(a, x)
        loss2 = torch.sum(c*f(a))
        cost = loss1 + loss2
        optimizer.zero_grad(set_to_none=True)
        cost.backward(retain_graph=True)
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
        
        print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

    attack_images = 1/2*(nn.Tanh()(w) + 1)
    
    return attack_images





def add_nois_attack(model, x_test):
    noise_intensity = 0.1
    X_noisy = x_test 
    y_pred = model.predict(x_test)
    count=sum(y_pred)
    total=len(y_pred)
    real_prediction = count/total
    attack_prediction = real_prediction
    #print()
    coef = sparse.csr.csr_matrix(model.coef_)
    X_noisy = X_noisy.multiply(coef)

    y_pred = model.predict(X_noisy)
    count=sum(y_pred)
    total=len(y_pred)
    attack_prediction = count/total
    # print("attack detection rate: ",attack_prediction)
    # print("real detection rate: ",real_prediction)
    # print("score change, if d > 0 it's good, if d < 0 t's bad")
    # print("delta: ", (real_prediction - attack_prediction))
    return y_pred
    









def main():
    with open('classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    print("finished open the file")
    
    #1219
    dim_bound = 10
    #y_pred = model.clf.predict(model.X_test)
    #evaluate_metrics(y_pred,model.X_test, model.y_test)
    
    #print("SNA")
    #pred_san = add_nois_attack(model.clf,model.X_test)
    #evaluate_metrics(pred_san,model.X_test, model.y_test)
    #img = torch.empty((0,model.X_test.shape[1]), dtype=torch.int64)
    attacked_image = None#sparse.csr_matrix(img,shape=img.shape)
    for i in range(0,901,53):
        print(i,")\n")
        temp = cw_l2_attack(model.clf,model.X_test[i:i+52],model.y_test[i:i+52],max_iter=10)
        #cprint(type(temp),"blue")
        attacked_image = sparse.vstack((attacked_image,sparse.csr_matrix(temp.detach().numpy(),shape=temp.shape)))
        #print("\n",attacked_image.shape[0],",",attacked_image.shape[1],"\n")
    #attacked_image = sparse.csr_matrix(attacked_image,shape=attacked_image.shape)
    
    
    y_pred = model.clf.predict(attacked_image)
    cprint(attacked_image.shape[0],"red")
    cprint(len(y_pred),"red")
    cprint("finshed prediction","blue")
    evaluate_metrics(y_pred, model.y_test[:len(y_pred)])

if __name__ == "__main__":
    main()