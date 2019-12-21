def conseguir_modelos():
  SVC=SVC(C=10, kernel='rbf',gamma='scale')
  OVO=OneVsOneClassifier(SVC(C=10, kernel='rbf',gamma='scale'))
  OVR=OneVsRestClassifier(SVC(C=10, kernel='rbf',gamma='scale'))
  ECOC=OutputCodeClassifier(SVC(C=10, kernel='rbf',gamma='scale'))
  return [SVC,OVO,OVR,ECOC]