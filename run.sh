for i in {0..4}
do
  python run_ac_offline.py --seed $i --env_name Ant --dataset expert --discrete_control 0 --state_dim 111 --action_dim 8 --tau 0.01 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  python run_ac_offline.py --seed $i --env_name Ant --dataset medexp --discrete_control 0 --state_dim 111 --action_dim 8 --tau 0.01 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  python run_ac_offline.py --seed $i --env_name Ant --dataset medium --discrete_control 0 --state_dim 111 --action_dim 8 --tau 0.5 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  python run_ac_offline.py --seed $i --env_name Ant --dataset medrep --discrete_control 0 --state_dim 111 --action_dim 8 --tau 0.5 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  python run_ac_offline.py --seed $i --env_name HalfCheetah --dataset expert --discrete_control 0 --state_dim 17 --action_dim 6 --tau 0.01 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  python run_ac_offline.py --seed $i --env_name HalfCheetah --dataset medexp --discrete_control 0 --state_dim 17 --action_dim 6 --tau 0.1 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  python run_ac_offline.py --seed $i --env_name HalfCheetah --dataset medium --discrete_control 0 --state_dim 17 --action_dim 6 --tau 0.33 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  python run_ac_offline.py --seed $i --env_name HalfCheetah --dataset medrep --discrete_control 0 --state_dim 17 --action_dim 6 --tau 0.5 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  python run_ac_offline.py --seed $i --env_name Hopper --dataset expert --discrete_control 0 --state_dim 11 --action_dim 3 --tau 0.01 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  python run_ac_offline.py --seed $i --env_name Hopper --dataset medexp --discrete_control 0 --state_dim 11 --action_dim 3 --tau 0.01 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  python run_ac_offline.py --seed $i --env_name Hopper --dataset medium --discrete_control 0 --state_dim 11 --action_dim 3 --tau 0.1 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  python run_ac_offline.py --seed $i --env_name Hopper --dataset medrep --discrete_control 0 --state_dim 11 --action_dim 3 --tau 0.5 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  python run_ac_offline.py --seed $i --env_name Walker2d --dataset expert --discrete_control 0 --state_dim 17 --action_dim 6 --tau 0.01 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  python run_ac_offline.py --seed $i --env_name Walker2d --dataset medexp --discrete_control 0 --state_dim 17 --action_dim 6 --tau 0.1 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  python run_ac_offline.py --seed $i --env_name Walker2d --dataset medium --discrete_control 0 --state_dim 17 --action_dim 6 --tau 0.33 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  python run_ac_offline.py --seed $i --env_name Walker2d --dataset medrep --discrete_control 0 --state_dim 17 --action_dim 6 --tau 0.5 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 &
  sleep 1
  wait
done
