algorithm = 'q_learning'

if algorithm == 'qdot_learning':
    from qdot_learning import QdotLearning
    learning = QdotLearning()
elif algorithm == 'q_learning':
    from q_learning import QLearning
    learning = QLearning()
elif algorithm == 'randomwalk':
    from randomwalk import RandomWalk
    learning = RandomWalk()

learning.learn()
