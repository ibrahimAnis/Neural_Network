#include<iostream>
#include<list>
#include <random>
#include<math.h>
#include<stdio.h>
#include<fstream>
using namespace std;
class Neuron;
class Dendrite 
{
private:
double weight;
Neuron *inputNeuron;
double deltaWeight;
public:
Dendrite() 
{
this->weight=0.0;
this->deltaWeight=0.0;
this->inputNeuron=NULL; 
}
Dendrite(const Dendrite &other) 
{
this->weight=other.weight;
this->deltaWeight=other.deltaWeight;
this->inputNeuron=other.inputNeuron; 
}
Dendrite & operator=(Dendrite other) 
{
this->weight=other.weight;
this->deltaWeight=other.deltaWeight;
this->inputNeuron=other.inputNeuron;
return *this;
}
virtual ~Dendrite() {}
void setWeight(double weight) 
{
this->weight=weight;
}
double getWeight() 
{
return this->weight; 
}
void setInputNeuron(Neuron *inputNeuron) 
{
this->inputNeuron=inputNeuron; 
}
Neuron * getInputNeuron() 
{
return this->inputNeuron; 
}
void setDeltaWeight(double deltaWeight)
{
this->deltaWeight=deltaWeight;
}
double getDeltaWeight()
{
return this->deltaWeight;
}
};
class Neuron 
{
private:
double output;
double error;
double gradient;
list<Dendrite *> dendrites;
public:
Neuron() 
{
this->output=1.0;
this->error=0.0; 
this->gradient=0.0;
}
Neuron(const Neuron &other) 
{
this->output=other.output;
this->error=other.error;
this->gradient=other.gradient;
this->dendrites=other.dendrites; 
}
virtual ~Neuron() {}
Neuron & operator=(Neuron other) 
{
this->output=other.output;
this->error=other.error;
this->gradient=other.gradient;
this->dendrites=other.dendrites;
return *this; 
}
void setOutput(double output) 
{
this->output=output;
}
double getOutput()
{
return this->output; 
}
void setError(double error) 
{
this->error=error; 
}
double getError() 
{
return this->error; 
}
void setGradient(double gradient)
{
this->gradient=gradient;
}
double getGradient()
{
return this->gradient;
}
list<Dendrite*>* getDendrites()
{
return &(this->dendrites);
}

void addDendrite(Dendrite *dendrite) 
{
this->dendrites.push_back(dendrite); }
Dendrite * getDendrite(int index)
{
list<Dendrite *>::iterator listIterator=dendrites.begin();
advance(listIterator,index);
return *listIterator; 
}
long getDendriteCount()
{
return dendrites.size();
}
};
class Network 
{
private:
list<list<Neuron *> *> layers;
double alpha,eta;
list<double> labels;
int labelsCount=0;
int featuresCount=0;
public:
Network()
{
this->eta=0.01;
this->alpha=0.1;
}
Network(const Network &other)
{
this->layers=other.layers;
this->eta=other.eta;
this->alpha=other.alpha;
this->labels=other.labels;
}
Network & operator=(Network other)
{
this->layers=other.layers;
this->eta=other.eta;
this->alpha=other.alpha;
this->labels=other.labels;
return *this; 
}
virtual ~Network() 
{
}
void train(string file)
{

ifstream f;
double x;
vector<vector<double>*> trainingSet;
//Vector v;
vector<double>* v;
f.open(file);
if(f.fail())
{
//cout<<"File not found"<<endl;
return;
}
int i=0;
v=new vector<double>();
while(1)
{
if(i==this->labelsCount+this->featuresCount)
{
i=0;
trainingSet.push_back(v);
v=new vector<double>();
}
f>>x;
if(f.fail()) break;
v->push_back(x);
++i;
}
Neuron* neuron;
Neuron* backLayerNeuron;
Dendrite* dendrite;
list<Neuron*>* layer;
list<list<Neuron*>*>::iterator layerIterator;
list<list<Neuron*>*>::reverse_iterator reverseItr1;
list<list<Neuron*>*>::reverse_iterator reverseItr2;
list<Neuron*>* previousLayer;
int backLayerNeuronCount=0;
double desigmoid=0.0;
double weight=0.0;
double deltaWeight=0.0;
double gradient=0.0;
int layerCount=this->layers.size();
int neuronCount;
int dendriteCount;
double output=0.0;
cout<<"Training starts...."<<endl;
double acc;
int inc=0;
while(1)
{
inc++;
acc=0.0;
int j;
for(int i=0;i<trainingSet.size();++i)
{
//if(inc%10000==0) cout<<"Training set"<<i<<endl;
v=trainingSet[i];
for(j=0;j<featuresCount;++j)
{
this->setInput(j,v->at(j));
//cout<<"Input"<<v->at(j)<<endl;
}
for(int k=0;k<labelsCount;k++,++j) 
{
this->setLabel(k,v->at(j));
//cout<<"output"<<v->at(j)<<endl;
}

layerIterator=this->layers.begin();
++layerIterator;
for(int i=2;i<=layerCount;i++,++layerIterator)
{
layer=*layerIterator;
neuronCount=layer->size();
//cout<<"Layer Number "<<i<<endl;
//cout<<"Neuron Count "<<neuronCount<<endl;
for(j=0;j<neuronCount;j++)
{
neuron=this->getNeuron(i,j);
dendriteCount=neuron->getDendriteCount();
//cout<<"dendriteCount:"<<dendriteCount<<endl;
output=0.0;
for(int k=0;k<dendriteCount;k++)
{
dendrite=neuron->getDendrite(k);
backLayerNeuron=dendrite->getInputNeuron();
//cout<<"backlayer neuron:"<<backLayerNeuron->getOutput()<<" dendrite weight:"<<dendrite->getWeight()<<endl;
output+=backLayerNeuron->getOutput()*dendrite->getWeight();
}
//cout<<"output after comp..."<<output<<endl;
if(dendriteCount!=0) output=(1.0/(1.0+exp(-1.0*output)));
else output=1.0;
//cout<<"Output of "<<j<<" "<<output<<endl;
neuron->setOutput(output);
}
}

//cout<<"Feed propogation Ends here"<<endl;

double err,error;
list<double>::iterator labelIterator;
labelIterator=this->labels.begin();
//cout<<"label size:"<<this->labels.size()<<endl;
int ii=0;
error=0.0;
while(labelIterator!=this->labels.end())
{
neuron=this->getNeuron(layerCount,ii);
//cout<<"Label value:"<<*labelIterator<<"neuron value:"<<neuron->getOutput()<<endl;
err=*labelIterator-neuron->getOutput();
//cout<<"Error for label "<<ii<<" is "<<err<<endl;
neuron->setError(err);
++labelIterator;
ii++;
error+=pow(err,2);
}

error/=2.0;
error=sqrt(error);
acc+=error;
//cout<<"Error computed"<<endl;
reverseItr1=layers.rbegin();
reverseItr2=layers.rbegin();
++reverseItr2;
list<Neuron*>::iterator neuronIterator;
list<Neuron*>::iterator previousLayerIterator;
list<Dendrite*>::iterator dendriteIterator;
list<Dendrite*> *dendrites;
for(int i=layerCount;i>1;i--)
{
//cout<<"Layer Number "<<i<<endl;
layer=*reverseItr1;
neuronCount=layer->size();
previousLayer=*reverseItr2;
neuronIterator=layer->begin();
for(int j=0;j<neuronCount;j++,++neuronIterator)
{
if(j==0 && i!=layerCount) continue;
neuron=(*neuronIterator);
desigmoid=neuron->getOutput()*(1.0-neuron->getOutput());
//cout<<"Desigmoid:"<<desigmoid<<endl;
gradient=neuron->getError()*desigmoid;
neuron->setGradient(gradient);
neuron->setError(0.0);
//cout<<"Neuron gradient "<<gradient<<endl;
backLayerNeuronCount=previousLayer->size();
dendrites=neuron->getDendrites();
//cout<<"dendrite size:"<<dendrites->size()<<endl;
dendriteIterator=dendrites->begin();
previousLayerIterator=previousLayer->begin();
for(int k=0;k<backLayerNeuronCount;k++)
{
backLayerNeuron=*(previousLayerIterator);
dendrite=*(dendriteIterator);
//cout<<"Dendrite weight Before:"<<dendrite->getWeight()<<endl;
deltaWeight=this->alpha*dendrite->getDeltaWeight()+this->eta*backLayerNeuron->getOutput()*neuron->getGradient();
dendrite->setDeltaWeight(deltaWeight);
dendrite->setWeight(dendrite->getDeltaWeight()+dendrite->getWeight());
//cout<<"Back layer neuron output:"<<backLayerNeuron->getOutput()<<endl;
error=(neuron->getGradient()*dendrite->getWeight())+backLayerNeuron->getError();
backLayerNeuron->setError(error);
//cout<<"Dendrite weight After:"<<dendrite->getWeight()<<endl;
//cout<<"Back Layer Neuron Error:"<<backLayerNeuron->getError()<<endl;
++previousLayerIterator;
++dendriteIterator;
}
}
++reverseItr1;
++reverseItr2;
}
}
cout<<"TOTAL NETWORK ERROR----------------------"<<acc<<endl;
if(acc<0.01) break;
}

// add weight to file
//cout<<"add weights to file"<<endl;
freopen("nn.bin","w",stdout);
int d,l_size;
layerIterator=layers.begin();
++layerIterator;
int layerNumber=2;
i=0;
while(layerIterator!=layers.end())
{
layer=*layerIterator;
l_size=layer->size();
for(int jj=0;jj<l_size;jj++)
{
neuron=this->getNeuron(layerNumber,jj);
d=neuron->getDendriteCount();
for(int kk=0;kk<d;kk++)
{
dendrite=neuron->getDendrite(kk);
cout<<dendrite->getWeight()<<endl;
//printf("Dendrite weight %d\n",dendrite->getWeight());
//cout<<"Dendrite Weight of"<<i<<" "<<dendrite->getWeight()<<endl;
}
}
++layerIterator;
++layerNumber;
}
//cout<<"weights added"<<endl;
fclose(stdout);
}
void setNumberOfLayers(int numberOfLayers) 
{
for(int x=0;x<numberOfLayers;x++) layers.push_back(new list<Neuron *>); 
}
void setNumberOfNeurons(int layerNumber,int numberOfNeurons)
{
if(layerNumber==this->layers.size())
{
//cout<<"initializing labels"<<endl;
for(int i=0;i<numberOfNeurons;i++) this->labels.push_back(0.0);
//cout<<"label size"<<this->labels.size()<<endl;
this->labelsCount=this->labels.size();
}
if(layerNumber==1) this->featuresCount=numberOfNeurons;
Neuron **neuronsOnPreviousLayer=NULL;
int numberOfNeuronsOnPreviousLayer;
if(layerNumber>1)
{
list<list<Neuron *> *>::iterator i=layers.begin();
advance(i,layerNumber-2);
list<Neuron *> *lastPopulatedLayer=*i;
numberOfNeuronsOnPreviousLayer=lastPopulatedLayer->size();
neuronsOnPreviousLayer=new Neuron *[numberOfNeuronsOnPreviousLayer];
list<Neuron *>::iterator ni=lastPopulatedLayer->begin();
int k=0;
while(ni!=lastPopulatedLayer->end())
{
neuronsOnPreviousLayer[k]=*ni;
k++;
++ni;
}
}
list<Neuron *> *layer;
list<list<Neuron *> *>::iterator layerIterator=layers.begin();
advance(layerIterator,layerNumber-1);
layer=*layerIterator;
Neuron *neuron;
Dendrite *dendrite;
for(int x=0;x<numberOfNeurons;x++)
{
neuron=new Neuron;
layer->push_back(neuron);
if(layerNumber>1 && (x!=0 || layerNumber==layers.size()))
{
for(int ss=0;ss<numberOfNeuronsOnPreviousLayer;ss++)
{
dendrite=new Dendrite;
dendrite->setInputNeuron(neuronsOnPreviousLayer[ss]);
neuron->addDendrite(dendrite);
}
}
}
if(neuronsOnPreviousLayer) delete [] neuronsOnPreviousLayer;
}
void setInput(int neuronIndex,double input) 
{
Neuron *neuron=this->getNeuron(1,neuronIndex);
neuron->setOutput(input);
}
void setLabel(int neuronIndex,double value) 
{
list<double>::iterator labelIterator=labels.begin();
advance(labelIterator,neuronIndex);
*labelIterator=value;
}
private:
Neuron * getNeuron(int layerNumber,int neuronIndex) 
{
//cout<<"Getting "<<neuronIndex<<endl;
list<list<Neuron *> *>::iterator layerIterator=layers.begin();
advance(layerIterator,layerNumber-1);
list<Neuron *> *layer=*(layerIterator);
list<Neuron *>::iterator neuronIterator=layer->begin();
advance(neuronIterator,neuronIndex);
//cout<<"Value of neuron:"<<neuronIndex<<" "<<(*neuronIterator)->getOutput()<<endl; 
return *neuronIterator; 
}
public:
void setWeights(void (*ptr)(int,double *)) // still pending 
{
int size;
list<list<Neuron *> *>::iterator layerIterator=this->layers.begin();
list<Neuron *> *layer;
int thisLayer,nextLayer;
int f,d;
size=0;
for(f=1;f<layers.size();f++) 
{
layer=*layerIterator;
thisLayer=layer->size();
++layerIterator;
layer=*layerIterator;
nextLayer=layer->size();
if(f!=layers.size()-1) nextLayer--;
size=size+(thisLayer*nextLayer); 
}
//cout<<"size:"<<size<<endl;
double* dataSet=new double[size];
ptr(size,dataSet);
layerIterator=layers.begin();
++layerIterator;
Neuron *neuron;
Dendrite* dendrite;
int layerNumber=2;
int i=0;
while(layerIterator!=layers.end())
{
layer=*layerIterator;
f=layer->size();
for(int jj=0;jj<f;jj++)
{
neuron=this->getNeuron(layerNumber,jj);
d=neuron->getDendriteCount();
for(int kk=0;kk<d;kk++)
{
dendrite=neuron->getDendrite(kk);
dendrite->setWeight(dataSet[i++]);
//cout<<"Dendrite Weight of"<<i<<" "<<dendrite->getWeight()<<endl;
}
}
++layerIterator;
++layerNumber;
}
delete [] dataSet;
}

void predict()
{
int size;
list<list<Neuron *> *>::iterator layerIterator=this->layers.begin();
list<Neuron *> *layer;
int thisLayer,nextLayer;
int f,d;

// Set predicted weights
//cout<<"Predicting weights"<<endl;
freopen("nn.bin","r",stdin);
layerIterator=layers.begin();
++layerIterator;
Neuron *neuron;
double output;
Dendrite* dendrite;
double wt;
int layerNumber=2;
int i=0,j;
int layerCount,neuronCount,dendriteCount;
Neuron* backLayerNeuron;
while(layerIterator!=layers.end())
{
layer=*layerIterator;
f=layer->size();
for(int jj=0;jj<f;jj++)
{
neuron=this->getNeuron(layerNumber,jj);
d=neuron->getDendriteCount();
for(int kk=0;kk<d;kk++)
{
dendrite=neuron->getDendrite(kk);
cin>>wt;
dendrite->setWeight(wt);
//cout<<"Dendrite Weight of"<<i<<" "<<dendrite->getWeight()<<endl;
i++;
}
}
++layerIterator;
++layerNumber;
}
fclose(stdin);


layerCount=this->layers.size();
layerIterator=this->layers.begin();
++layerIterator;
for(int i=2;i<=layerCount;i++,++layerIterator)
{
layer=*layerIterator;
neuronCount=layer->size();
//cout<<"Layer Number "<<i<<endl;
//cout<<"Neuron Count "<<neuronCount<<endl;
for(j=0;j<neuronCount;j++)
{
neuron=this->getNeuron(i,j);
dendriteCount=neuron->getDendriteCount();
//cout<<"dendriteCount:"<<dendriteCount<<endl;
output=0.0;
for(int k=0;k<dendriteCount;k++)
{
dendrite=neuron->getDendrite(k);
backLayerNeuron=dendrite->getInputNeuron();
//cout<<"backlayer neuron:"<<backLayerNeuron->getOutput()<<" dendrite weight:"<<dendrite->getWeight()<<endl;
output+=backLayerNeuron->getOutput()*dendrite->getWeight();
}
//cout<<"output after comp..."<<output<<endl;
if(dendriteCount!=0) output=(1.0/(1.0+exp(-1.0*output)));
else output=1.0;
//cout<<"Output of "<<j<<" "<<output<<endl;
neuron->setOutput(output);
}
}




}
double getOutput(int index)
{
int layerCount=this->layers.size();
return this->getNeuron(layerCount,index)->getOutput();
}

};
void weightGenerator(int size,double *dataSet)
{
std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0,1.0);
for(int i=0;i<size;++i)
{
dataSet[i]=distribution(generator);
//cout<<dataSet[i]<<endl;
}
}
int main()
{
Network network;
network.setNumberOfLayers(4);
network.setNumberOfNeurons(1,2); // we are referring first layer with 1
network.setNumberOfNeurons(2,5);
network.setNumberOfNeurons(3,4);
network.setNumberOfNeurons(4,1);
network.setWeights(weightGenerator);
network.train(string("train.data"));
network.setInput(0,0.0);
network.setInput(1,1.0);
network.predict();
double a=network.getOutput(0);
cout<<"XOR of 0.0 and 1.0 is "<<a<<endl;
return 0;
}