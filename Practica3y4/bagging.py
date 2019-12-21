def conseguir_modelos():
  knn=BaggingClassifier(KNeighborsClassifier(10)) 
  svm=BaggingClassifier(SVC(C=10, kernel='rbf',gamma='scale'))
  tree=BaggingClassifier(DecisionTreeClassifier())
  return [knn,svm,tree]