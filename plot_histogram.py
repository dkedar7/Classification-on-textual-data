#importing datasets
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt

#defining the subclasses (8, in total, belonging to the 2 'parent' classes) to be used in this project
computer_technology_subclasses=[['comp.graphics'],['comp.os.ms-windows.misc'],['comp.sys.ibm.pc.hardware'],['comp.sys.mac.hardware']]
recreational_activity_subclasses=[['rec.autos'],['rec.motorcycles'],['rec.sport.baseball'],['rec.sport.hockey']]
        
#Plotting required histograms
def plot_histogram(target_set):
    pretty_print={'all': '(all subsets):','train': '(only training subsets)','test': '(only testing subsets)'}
    print('')
    print('Number of documents per topic '+pretty_print[target_set])
    
    #initialisation
    no_of_documents=[]
    no_of_documents_in_comp_tech=0
    no_of_documents_in_rec_act=0
    
    #filling in all the subclasses with documents (in a random fashion), and getting the number of documents in each class
    for i in range(4):
        number=len(fetch_20newsgroups(subset=target_set,categories=computer_technology_subclasses[i],shuffle=True,random_state=42).data)
        no_of_documents.append(number)
        no_of_documents_in_comp_tech+=number 
    for i in range(4):
        number=len(fetch_20newsgroups(subset=target_set,categories=recreational_activity_subclasses[i],shuffle=True,random_state=42).data)
        no_of_documents.append(number)
        no_of_documents_in_rec_act+=number
    subclasses_of_documents=computer_technology_subclasses+recreational_activity_subclasses
    
    for i in range(8):
        spaces=''
        for j in range(26-len(subclasses_of_documents[i][0])):
            spaces+=' '
        print(spaces+subclasses_of_documents[i][0]+' : '+str(no_of_documents[i]))
    print('')
    
    print('Computer Technology documents: '+str(no_of_documents_in_comp_tech))
    print('Recreational Activity: '+str(no_of_documents_in_rec_act))

    #Plotting the required histograms
    print('')
    print('Histogram of the number of documents per topic '+pretty_print[target_set])
    
    x_labels=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
    
    #feeding in the parameters to plot the histogram
    fig,ax=plt.subplots()
    ax.set_xticks([i+0.5 for i in range(1,9)])
    ax.set_xticklabels(x_labels,rotation=60,ha='right',fontsize=10)
    
    rects=plt.bar([i for i in range(1,9)],no_of_documents,0.5,align='edge',color='red')
    plt.xlabel('Topic Name',fontsize=10)
    plt.ylabel('Number of Documents',fontsize=10)
    plt.title('Number of documents per topic '+pretty_print[target_set],fontsize=15)
    plt.axis([0.5,9,0,1100])
    
    for rect in rects:
        height=rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2.,1*height,'%d' % int(height),ha='center',va='bottom')
    
    plt.show()

#plotting histogram for the whole document corpus
plot_histogram('all')

#plotting histogram for the training corpus
plot_histogram('train')

#plotting histogram for the testing corpus
plot_histogram('test')
