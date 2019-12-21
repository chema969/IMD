def conseguir_modelos():
  svm_model = SVC(kernel='rbf')

  Cs = np.logspace(-5, 15, num=6, base=2)
  Gs = np.logspace(-15, 8, num=6, base=2)
  optimo = GridSearchCV(estimator=svm_model, param_grid=dict(C=Cs,gamma=Gs),
  n_jobs=-1,cv=5)
  return [optimo]