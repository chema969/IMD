def conseguir_modelos():
  KNN=KNeighborsClassifier(10)
  OVO=OneVsOneClassifier(KNeighborsClassifier(10))
  OVR=OneVsRestClassifier(KNeighborsClassifier(10))
  ECOC=OutputCodeClassifier(KNeighborsClassifier(10))
  return [KNN,OVO,OVR,ECOC]