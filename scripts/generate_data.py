import os
import sys
import random
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--exp", type=int, choices=[1,2,3], help="Experiment No")
    parser.add_argument("--scenario", type=int, choices=[1,2,3], help="Scenario No")
    parser.add_argument("--data_size", type=int, help="Data Size")
    parser.add_argument("--mode", type=str, choices=['train', 'test'], help="Train or Test")
    parser.add_argument("--seed", type=int, help="Random Seed", required=True)
    parser.add_argument("--snr", type=float, help="SNR (dB)")
    parser.add_argument("--T", type=int, help="The Number of Snapshots")
    
    args = parser.parse_args()
    
    print(f"\nExecuting data generation for experiment {args.exp} - scenario {args.scenario} - {args.mode} mode - snr {args.snr} - T {args.T}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    snr = args.snr
    T = args.T
    
    data = []
    save_idx = 0
    fs = 10000
    
    if snr == None:
        snr_random = True
    else:
        snr_random = False
    if T == None:
        T_random = True
    else:
        T_random = False

    for i in range(args.data_size):
        print(f"Generating sample {i}/{args.data_size}")

        if snr_random:
            snr = -20 + 20*np.random.rand()
            
        if T_random:
            T = np.random.randint(low=100, high=1000)

        t = np.arange(1/fs, (T+1)/fs, 1/fs)[:T]
        
        # sensor positions
        if args.exp == 1: # different sparse array types
            if args.scenario == 1: # minimum redundant array
                M = 7
                sensor_pos = np.zeros((M,3))
                sensor_pos[:,0] = np.array([1,2,3,6,10,14,16])*0.5
                sensor_pos_ula = np.zeros((16,3))
                sensor_pos_ula[:,0] = np.arange(1,17)*0.5
                sensor_pos_intact = np.zeros((M,3))
                sensor_pos_intact[:,0] = np.array([1,2,3,6,10,14,16])*0.5
                
            elif args.scenario == 2: # nested array
                M = 8
                sensor_pos = np.zeros((M,3))
                sensor_pos[:,0] = np.array([1,2,3,4,5,6,11,16])*0.5
                sensor_pos_ula = np.zeros((16,3))
                sensor_pos_ula[:,0] = np.arange(1,17)*0.5
                sensor_pos_intact = np.zeros((M,3))
                sensor_pos_intact[:,0] = np.array([1,2,3,4,5,6,11,16])*0.5
                
            elif args.scenario == 3: # coprime array
                M = 8
                sensor_pos = np.zeros((M,3))
                sensor_pos[:,0] = np.array([1,5,6,9,11,13,16,17])*0.5
                sensor_pos_ula = np.zeros((17,3))
                sensor_pos_ula[:,0] = np.arange(1,18)*0.5
                sensor_pos_intact = np.zeros((M,3))
                sensor_pos_intact[:,0] = np.array([1,5,6,9,11,13,16,17])*0.5
                
        elif args.exp == 2: # sensor malfunctions
            if args.mode == "train": # single training data for both scenario
                M = np.random.randint(low=5, high=9)
                sensor_pos = np.zeros((M,3))
                sensor_pos[:,0] = np.random.choice(np.array([1,2,3,4,5,6,11,16]), M, replace=False)*0.5
                sensor_pos[:,0] = np.sort(sensor_pos[:,0])
                sensor_pos_ula = np.zeros((16,3))
                sensor_pos_ula[:,0] = np.arange(1,17)*0.5
                sensor_pos_intact = np.zeros((8,3))
                sensor_pos_intact[:,0] = np.array([1,2,3,4,5,6,11,16])*0.5
                
            elif args.mode == "test":
                if args.scenario == 1: # intact array
                    M = 8
                    sensor_pos = np.zeros((M,3))
                    sensor_pos[:,0] = np.array([1,2,3,4,5,6,11,16])*0.5
                    sensor_pos_ula = np.zeros((16,3))
                    sensor_pos_ula[:,0] = np.arange(1,17)*0.5
                    sensor_pos_intact = np.zeros((M,3))
                    sensor_pos_intact[:,0] = np.array([1,2,3,4,5,6,11,16])*0.5
                    
                elif args.scenario == 2: # faulty array
                    M = np.random.randint(low=5, high=8)
                    sensor_pos = np.zeros((M,3))
                    sensor_pos[:,0] = np.random.choice(np.array([1,2,3,4,5,6,11,16]), M, replace=False)*0.5
                    sensor_pos[:,0] = np.sort(sensor_pos[:,0])
                    sensor_pos_ula = np.zeros((16,3))
                    sensor_pos_ula[:,0] = np.arange(1,17)*0.5
                    sensor_pos_intact = np.zeros((M,3))
                    sensor_pos_intact[:,0] = np.array([1,2,3,4,5,6,11,16])*0.5
        
        elif args.exp == 3: # unknown number of sources
            M = np.random.randint(low=5, high=9)
            sensor_pos = np.zeros((M,3))
            sensor_pos[:,0] = np.random.choice(np.array([1,2,3,4,5,6,11,16]), M, replace=False)*0.5
            sensor_pos[:,0] = np.sort(sensor_pos[:,0])
            sensor_pos_ula = np.zeros((16,3))
            sensor_pos_ula[:,0] = np.arange(1,17)*0.5 
            sensor_pos_intact = np.zeros((M,3))
            sensor_pos_intact[:,0] = np.array([1,2,3,4,5,6,11,16])*0.5
           
        # source signal
        if args.exp == 1 or args.exp == 2: # different sparse array types / sensor malfunctions
            N = 4
        elif args.exp == 3: # unknown number of sources
            N = np.random.randint(low=1, high=5)
        source_var = 1
        source_phi = np.zeros((N,1))
        for j in range(N):
            cand = 30 + 120*np.random.rand()
            while np.any(abs(cand - source_phi[:j]) < 10):
                cand = 30 + 120*np.random.rand()
            source_phi[j] = cand
        source_the = 90*np.ones((N,1))
        s = np.zeros((N, len(t)), dtype='complex_')
        for n in range(N):
            s[n,:] = np.sqrt(source_var) * (np.random.randn(1,len(t)) + 1j*np.random.randn(1,len(t)))

        # noise
        noise_var = source_var/(10**(snr/10))
        v = np.sqrt(noise_var)*np.random.randn(M,T) + 1j*np.sqrt(noise_var)*np.random.randn(M,T)

        # received signal
        A = np.exp(-2j*np.pi*(sensor_pos[:,0] * np.cos(np.deg2rad(source_phi)) * np.sin(np.deg2rad(source_the)) + 
                              sensor_pos[:,1] * np.sin(np.deg2rad(source_phi)) * np.sin(np.deg2rad(source_the)) +
                              sensor_pos[:,2] * np.cos(np.deg2rad(source_the))
                              )).T
        A_ula = np.exp(-2j*np.pi*(sensor_pos_ula[:,0] * np.cos(np.deg2rad(source_phi)) * np.sin(np.deg2rad(source_the)) + 
                                  sensor_pos_ula[:,1] * np.sin(np.deg2rad(source_phi)) * np.sin(np.deg2rad(source_the)) +
                                  sensor_pos_ula[:,2] * np.cos(np.deg2rad(source_the))
                                  )).T      
        A_intact = np.exp(-2j*np.pi*(sensor_pos_intact[:,0] * np.cos(np.deg2rad(source_phi)) * np.sin(np.deg2rad(source_the)) + 
                                     sensor_pos_intact[:,1] * np.sin(np.deg2rad(source_phi)) * np.sin(np.deg2rad(source_the)) +
                                     sensor_pos_intact[:,2] * np.cos(np.deg2rad(source_the))
                                     )).T
        
        y = A@s + v

        # covariance matrix
        cm = (y@y.conj().T)/y.shape[1]
        cm_true = A @ np.diag(np.full(N,1)) @ A.conj().T
        cm_ula_true = A_ula @ np.diag(np.full(N,1)) @ A_ula.conj().T
        cm_intact_true = A_intact @ np.diag(np.full(N,1)) @ A_intact.conj().T
        
        data.append({'signals':y, 'cm':cm, 'cm_true':cm_true, 'cm_ula_true':cm_ula_true, 'cm_intact_true':cm_intact_true, 'label':source_phi, 'sensor_pos':sensor_pos})      

        
    if args.mode == "train":
        np.save(f"../data/experiment_{args.exp}/scenario_{args.scenario}/data_train.npy", data)
    elif args.mode == "test":
        np.save(f"../data/experiment_{args.exp}/scenario_{args.scenario}/data_test_snr{int(snr)}_t{T}.npy", data)

           
