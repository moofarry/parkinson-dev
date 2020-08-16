import h5py
import numpy as np
import time
from time import time

from sklearn import svm, datasets
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.model_selection import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.svm import *

from scipy import interp
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.naive_bayes import *
from sklearn.neighbors import KNeighborsClassifier
######################################################################        
txt=  '.txt'   
directorio= '/home/robotica/Dropbox/pd_database/am-fm/'
izq_train= '/home/robotica/Dropbox/pd_database/cv/train' 
izq_test= '/home/robotica/Dropbox/pd_database/cv/test' 
target_names=['HC','PD']
######################################################################
palabra='pakatak    n'
caracteristica='mhec'
random_seed=None
n_components=62
'''
palabra=input('Palabra: ')
caracteristica= input('Caracterista: ')
ran_seed = input('Randon_seed: ')   #42
n_comp= input('n_components: ')   #62
random_seed= int(ran_seed)
n_components= int(n_comp)
'''

######################################################################
tiempos=[]
svmk=[]
tprs = []	
aucs = []
mean_fpr = np.linspace(0, 1, 100)
######################################################################
tn_=[]
fp_=[]
fn_=[]
tp_=[]
######################################################################
for i in range(10):
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    Kappa=[]
    X0=[]
    X1=[]
    target0=[]
    target1=[]
    print('\n')    
    print('Validacion Cruzada= ', i+1)
    Train= izq_train +str(i)+txt  
    Test= izq_test +str(i)+txt 
    print(type(Train))
    hf_train=open(Train,'r')
    hf_test= open(Test,'r')
    lines_train=hf_train.readlines()
    lines_test =hf_test.readlines()
    hf_test.close()
    hf_train.close()
    ######################################################################
    #Palabras [inicio, tipo_file]
    ka=['ka-ka-ka/','_ka.h5']
    pakata=['pakata/','_pakata.h5']
    pa=['pa-pa-pa/','_pa.h5']
    pataka=['pataka/','_pataka.h5']
    petaka=['petaka/','_petaka.h5']
    ta=['ta-ta-ta/','_ta.h5']
    
    
    for elem in lines_train:
        archivo,label= elem.split(',')
       # archivo=directorio+palabra+archivo+palabra
        if palabra =='ka':
            archivo=directorio+ka[0]+archivo+ka[1]
        elif palabra =='pakata':
            archivo=directorio+pakata[0]+archivo+pakata[1]
        elif palabra =='pa':
            archivo=directorio+pa[0]+archivo+pa[1]
        elif palabra =='pataka':
            archivo=directorio+pataka[0]+archivo+pataka[1]
        elif palabra =='petaka':
            archivo=directorio+petaka[0]+archivo+petaka[1]
        else:
            archivo=directorio+ta[0]+archivo+ta[1]

        hf_train = h5py.File(archivo,'r')
        
        data = hf_train.get(caracteristica)
        data = np.array(data).T
        FEAT_COV=np.cov(data.transpose())
        p=FEAT_COV.shape[0] 
        UPFEATS=FEAT_COV[np.triu_indices(p, k=0)]
        #acumular en listas el vector y el label o target
        X0.append(UPFEATS)
        target0.append(int(label))

    #finalizado el proceso para todos los archivos apilar en np.array    
    X_train=np.array(X0)
    y_train=np.array(target0)   
    
    for elem1 in lines_test:
        archivo1,label1= elem1.split(',')
       # archivo=directorio+palabra+archivo+palabra
        if palabra =='ka':
            archivo1=directorio+ka[0]+archivo1+ka[1]
        elif palabra =='pakata':
            archivo1=directorio+pakata[0]+archivo1+pakata[1]
        elif palabra =='pa':
            archivo1=directorio+pa[0]+archivo1+pa[1]
        elif palabra =='pataka':
            archivo1=directorio+pataka[0]+archivo1+pataka[1]
        elif palabra =='petaka':
        
            archivo1=directorio+petaka[0]+archivo1+petaka[1]
        else:
            archivo1=directorio+ta[0]+archivo1+ta[1]

        hf_test = h5py.File(archivo1,'r')
        data1 = hf_test.get(caracteristica)
        data1 = np.array(data1).T
        FEAT_COV1=np.cov(data1.T)
        p1=FEAT_COV1.shape[0] 
        UPFEATS1=FEAT_COV1[np.triu_indices(p1, k=0)]
        #acumular en listas el vector y el label o target
        X1.append(UPFEATS1)
        target1.append(int(label1))
        
    #finalizado el proceso p  
    X_test=np.array(X1)
    y_test=np.array(target1)
    #print(X_test.shape,y_test.shape)
    #print(X_train.shape,y_train.shape)
    #print('\n')

    #################################################################################
    #min and max scale
    scaler = MinMaxScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.fit_transform(X_test)
    #################################################################################

    pca = PCA(n_components=n_components, copy=True, whiten=False, svd_solver='auto', tol=0.0,               iterated_power='auto', random_state=random_seed).fit(X_train)
    X_train= pca.transform(X_train)                                                                          
    X_test = pca.transform(X_test)

    #IMPLEMENTE SU CLASIFICADOR
    start_time = time()  
    ###################################################################
    model = KNeighborsClassifier(n_jobs=-1)
    
    params = {'n_neighbors':[5,7,9],
          'leaf_size':[5,7],
          'weights':['uniform', 'distance'],
          'algorithm':['auto'],
          'n_jobs':[-1]}
     

    clf = GridSearchCV(model, param_grid=params, n_jobs=1,cv=5)
    clf= clf.fit(X_train,y_train)
    print("Best Hyper Parameters:\n",clf.best_params_)
        

    ################################################################### 
    #ENTRENANDO   
    #
    probas_ = clf.fit(X_train, y_train).predict_proba(X_train)
    y_true, y_pred = y_test, clf.predict(X_test)
    report=classification_report(y_true, y_pred)
    acc=accuracy_score(y_true, y_pred)
    
    print('\n')
    elapsed_time = time() - start_time
    print("Elapsed time: %.10f seconds." % elapsed_time)
    tiempos.append(elapsed_time)
    print('\n')
    
    # Compute ROC curve and area the cusrve
    fpr, tpr, thresholds = roc_curve(y_train, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    

    
    tn, fp, fn, tp= confusion_matrix(y_true, y_pred).ravel()
    tn_.append(tn)
    fp_.append(fp)
    fn_.append(fn)
    tp_.append(tp)
    matrix= [['True Negative ', tn],
                 ['False Positive' , fp],
                 ['False Positive' , fn],
                 ['True Positive' , tp],
                 ['Sensitivity' , (tp/(tp+fn))],
                 ['Specificity,' , (tn/(tn+fp))]]    
    est=    [['Cross Validation' ,i],
             ['Accuracy Score ', acc]] 
        
   

    kappa=cohen_kappa_score(y_true, y_pred)
    Kappa.append(kappa)
    kapp =[[' Cohen’s kappa',kappa]]
    '''
    print(report)
    print(tabulate(matrix))
    print(tabulate(kapp))
    print(tabulate(est))

    print()
    svmk.append(acc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC CV %d (AUC = %0.2f)' % (i, roc_auc))
    '''
   
 
print('\n')
print('----------> Ponderado <----------')

print('\n')
print('tiempos acumulado', tiempos)
tiempos=np.array(tiempos)
promtiempos=np.mean(tiempos)
print('\n')
print('tiempo promdio',promtiempos)
print(caracteristica)
print('\n')

svmk=np.array(svmk)
desv0=np.std(svmk)
prom0=np.mean(svmk)
kapita=np.mean(Kappa)

tn_=np.array(tn_)
fp_=np.array(fp_)
fn_=np.array(fn_)
tp_=np.array(tp_)

prom_tn=np.mean(tn_)
prom_fp=np.mean(fp_)
prom_fn=np.mean(fn_)
prom_tp=np.mean(tp_)

desv_tn=np.std(tn_)
desv_fp=np.std(fp_)
desv_fn=np.std(fn_)
desv_tp=np.std(tp_)

prom_sensitivity=(prom_tp/(prom_tp+prom_fn))
desv_sensitivity=(desv_tp/(desv_tp+desv_fn))
prom_specificityi= (prom_tn/(prom_tn+prom_fp))
desv_specificity=(desv_tp/(desv_tp+desv_fn))


final=      [
            ['mean false negative (FN), ',prom_tn],
            ['desv false negative (FN), ', desv_tn],

            ['mean false positive (FP), ',prom_fp],
            ['desv false positive (FP), ', desv_fp],

            ['mean false negative (FN), ',prom_fn],
            ['desv false negative (FN), ', desv_fn],

            ['mean true positive (TP), ' ,prom_tp],
            ['desv true positive (TP), ', desv_tp],
            [],
            ['mean Sensitivity, ' , prom_sensitivity],
            ['desv Sensitivity, ' , desv_sensitivity],
            
            ['mean Specificity, ' ,prom_specificityi],            
            ['desv Specificity, ' ,desv_specificity],
            [],
            ['Deviation, ' ,desv0],
            ['Mean, ', prom0],     
    ]
    
          
#print(tabulate(final,)) #tablefmt='fancy_grid'

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$  std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC para la palabra '+palabra+' y Caracteristica '+caracteristica )
plt.legend(loc="lower right")
#plt.show()
