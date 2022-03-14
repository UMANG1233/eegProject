import numpy as np

def fComputeSlopeEntropy(vectorTimeSeries,iEmbeddedDimension,fGamma,fDelta):
    bFound=False
    fSlopEn=0.0
    p=0.0
    fSlope=0.0
    vectorSlopePattern=[]
    listPatternsFound_int=[]
    listPatternsFound_vector=[]
    
    for j in range(0,len(vectorTimeSeries)-(iEmbeddedDimension-1)):
        vectorSlopePattern.clear()
        for i in range(j+1,j+iEmbeddedDimension):
            fSlope=vectorTimeSeries[i]-vectorTimeSeries[i-1]
            if abs(fSlope)<=fDelta:
                vectorSlopePattern.append(0)
            elif fSlope>fDelta and fSlope<=fGamma:
                vectorSlopePattern.append(1)
            elif fSlope>fGamma:
                vectorSlopePattern.append(2)
            elif fSlope<-fDelta and fSlope>=-fGamma:
                vectorSlopePattern.append(-1)
            else :  
                vectorSlopePattern.append(-2)
        
        bFound=False
        
        for i in range(0,len(listPatternsFound_vector)):
            if listPatternsFound_vector[i]==vectorSlopePattern:
                listPatternsFound_int[i]=listPatternsFound_int[i]+1
                bFound = True
                break                                                                                                                                                                                                                   
        
        if bFound==False:
            listPatternsFound_int.append(1)
            listPatternsFound_vector.append(vectorSlopePattern)
    
    for i in range(0,len(listPatternsFound_vector)):
        p = listPatternsFound_int[i]/len(listPatternsFound_int)
        fSlopEn =fSlopEn -p*np.log2(p)
    print(listPatternsFound_int)
    return fSlopEn