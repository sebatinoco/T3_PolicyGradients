for i in {1..3}
do
    # CartPole
    python train_agent.py --env CartPole-v1 --training_iterations 100 --lr 0.005 --gamma 0.99 --batch_size 500 --use_baseline False --reward_to_go False --exp_name CartPole_exp_11_${i} # exp_11
    python train_agent.py --env CartPole-v1 --training_iterations 100 --lr 0.005 --gamma 0.99 --batch_size 500 --use_baseline False --reward_to_go True --exp_name CartPole_exp_21_${i} # exp_21
    python train_agent.py --env CartPole-v1 --training_iterations 100 --lr 0.005 --gamma 0.99 --batch_size 500 --use_baseline True --reward_to_go False --exp_name CartPole_exp_31_${i} # exp_31
    python train_agent.py --env CartPole-v1 --training_iterations 100 --lr 0.005 --gamma 0.99 --batch_size 500 --use_baseline True --reward_to_go True --exp_name CartPole_exp_41_${i} # exp_41

    python train_agent.py --env CartPole-v1 --training_iterations 100 --lr 0.005 --gamma 0.99 --batch_size 5000 --use_baseline False --reward_to_go False --exp_name CartPole_exp_12_${i} # exp_12
    python train_agent.py --env CartPole-v1 --training_iterations 100 --lr 0.005 --gamma 0.99 --batch_size 5000 --use_baseline False --reward_to_go True --exp_name CartPole_exp_22_${i} # exp_22
    python train_agent.py --env CartPole-v1 --training_iterations 100 --lr 0.005 --gamma 0.99 --batch_size 5000 --use_baseline True --reward_to_go False --exp_name CartPole_exp_32_${i} # exp_32
    python train_agent.py --env CartPole-v1 --training_iterations 100 --lr 0.005 --gamma 0.99 --batch_size 5000 --use_baseline True --reward_to_go True --exp_name CartPole_exp_42_${i} # exp_42

    # Pendulum
    python train_agent.py --env Pendulum-v1 --training_iterations 1000 --lr 0.005 --gamma 0.99 --batch_size 5000 --use_baseline False --reward_to_go False --exp_name Pendulum_exp_12_${i} # exp_12
    python train_agent.py --env Pendulum-v1 --training_iterations 1000 --lr 0.005 --gamma 0.99 --batch_size 5000 --use_baseline False --reward_to_go True --exp_name Pendulum_exp_22_${i} # exp_22
    python train_agent.py --env Pendulum-v1 --training_iterations 1000 --lr 0.005 --gamma 0.99 --batch_size 5000 --use_baseline True --reward_to_go False --exp_name Pendulum_exp_32_${i} # exp_32
    python train_agent.py --env Pendulum-v1 --training_iterations 1000 --lr 0.005 --gamma 0.99 --batch_size 5000 --use_baseline True --reward_to_go True --exp_name Pendulum_exp_42_${i} # exp_42
done