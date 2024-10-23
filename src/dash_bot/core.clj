(ns dash-bot.core
  (:import (org.encog.engine.network.activation ActivationSigmoid)
           (org.encog.ml.data.basic BasicMLDataSet)
           (org.encog.neural.networks BasicNetwork)
           (org.encog.neural.networks.layers BasicLayer)
           (org.encog.neural.networks.training.propagation.back Backpropagation)))

(def xor-input [[0.0 0.0] [1.0 0.0] [0.0 1.0] [1.0 1.0]])
(def xor-ideal [[0.0] [1.0] [1.0] [0.0]])

(defn vec-to-double-2d-array [v]
  (into-array (map double-array v)))

(defn create-network []
  (doto (BasicNetwork.)
    (.addLayer (BasicLayer. nil true 2))
    (.addLayer (BasicLayer. (ActivationSigmoid.) true 2))
    (.addLayer (BasicLayer. (ActivationSigmoid.) false 1))
    (-> .getStructure .finalizeStructure)
    (.reset)))

(defn train-network [network training-set]
  (let [train (Backpropagation. network training-set 0.7 0.3)]
    (loop [epoch 1]
      (.iteration train)
      (when (> (.getError train) 0.001)                     ;when is a macro of "if do"
        (println (str "Epoch #" epoch " Error:" (.getError train)))
        (recur (inc epoch))))
    (println )
    ))

(defn test-network [network training-set]
  (println "Neural Network Results:")
  (doseq [pair training-set]
    (let [input (.getInput pair)
          output (.compute network input)
          ideal (.getIdeal pair)]
      (println (str (.getData input 0) "," (.getData input 1)
                    ", actual=" (.getData output 0)
                    ",ideal=" (.getData ideal 0))))))

(defn -main []
  (let [network (create-network)]
    (let [training-set (BasicMLDataSet. (vec-to-double-2d-array xor-input) (vec-to-double-2d-array xor-ideal))]
      (train-network network training-set)
      (test-network network training-set))))