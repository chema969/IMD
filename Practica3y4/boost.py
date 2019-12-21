def conseguir_modelos():
  gboost=GradientBoostingClassifier(n_estimators=100)
  
  aboost=AdaBoostClassifier(n_estimators=100)
  return [gboost,aboost]