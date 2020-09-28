algorithm = 'qdot_learning'

if algorithm == 'qdot_learning':
    from qdot_learning import QdotLearning
    learning = QdotLearning()
elif algorithm == 'q_learning':
    from q_learning import QLearning
    learning = QLearning()
elif algorithm == 'randomwalk':
    from randomwalk import RandomWalk
    learning = RandomWalk()
elif algorithm == 'q_learning_rbf':
    from q_learning_rbf import QLearning_RBF
    learning = QLearning_RBF()
elif algorithm == 'qdot_learning_rbf':
    from qdot_learning_rbf import QdotLearning_RBF
    learning = QdotLearning_RBF()

learning.learn()
