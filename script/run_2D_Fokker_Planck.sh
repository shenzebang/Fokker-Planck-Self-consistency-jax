python main_PINN.py --test_frequency 100 --learning_rate 2e-3 --PDE 2D-Fokker-Planck --train_batch_size 64\
            --total_evolving_time 3. --ODE_tolerance 1e-5 --plot_domain_size 10. --diffusion_coefficient 1.\
            --method PINN


python main_COLT.py --test_frequency 100 --learning_rate 2e-3 --PDE 2D-Fokker-Planck --train_batch_size 64\
            --total_evolving_time 3. --ODE_tolerance 1e-5 --plot_domain_size 10. --diffusion_coefficient 1.\
            --method NWGF