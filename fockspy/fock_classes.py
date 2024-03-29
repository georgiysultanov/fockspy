#!/usr/bin/env python
# coding: utf-8

# In[1]:

#ver '0.1.18'

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse import coo_array
from scipy.sparse import identity
from scipy.sparse import diags



class fock_space:
    def __init__(self, about_excitations, global_exc, statistics, fermi_sea_modes = None):

        self.global_exc = global_exc #internal parameter
        self.statistics = statistics #internal parameter
            
        if statistics == 'Bose':
            
            if type(about_excitations) is int:
        #1)
                self.modes = about_excitations
        #2)
                self.local_exc = np.full(self.modes, global_exc)
            else:
        #1)
                self.modes = len(about_excitations)
        #2)
                self.local_exc = np.array(about_excitations)
        
        elif statistics == 'Fermi':
            if type(about_excitations) is int:
        #1)
                self.modes = about_excitations
        #2)
                self.local_exc = np.full(self.modes, 1)
            else:
        #1)
                self.modes = len(about_excitations)
        #2)
                self.local_exc = np.array(about_excitations)
        
        elif statistics == 'Fermi_sea':
            if type(about_excitations) is int:
        #1)
                self.modes = about_excitations
        #2)
                self.local_exc = np.full(self.modes, 1)
            else:
        #1)
                self.modes = len(about_excitations)
        #2)
                self.local_exc = np.array(about_excitations)
            
            self.fermi_sea_modes = fermi_sea_modes
                
                
        else:
            print('Сhoose the statistics: Bose, Fermi or Fermi_sea') 
        
        self.local_exc1 = np.array(self.local_exc+1) #internal parameter
        
        
        #3)
        if statistics == 'Fermi_sea':
            self.states_list = list(self.sea_generator())
        else:
            self.states_list = list(self.states_generator())
        
        #4)
        self.find_index = {state: idx for idx, state in enumerate(self.states_list)}
            
        
        #5)
        self.dimension  = len(self.states_list)
        
        #6)
        self.emptyH = coo_matrix((self.dimension , self.dimension ), dtype = complex)
        
        #7)
        self.eye = sparse.eye(self.dimension )
        
        
        if statistics == 'Bose':
            
        #8)    
            self.annihilate=[]
            current_state = []
            for k in range(self.modes):
                row = np.zeros(self.dimension )
                col = np.arange(self.dimension , dtype=int)
                data = np.zeros(self.dimension )
                for i in range(self.dimension ):
                    if self.states_list[i][k]==0:
                        row[i] = i
                    else:
                        current_state = list(self.states_list[i])
                        current_state[k]-=1
                        p = self.index(current_state)
                        row[i] = p
                        data[i]=(self.states_list[i][k])**0.5
                A = coo_matrix((data, (row, col)), shape=(self.dimension , self.dimension ), dtype = complex)
                self.annihilate.append(A) 
                
            
        #9)
            self.create=[]
            current_state = []
            for k in range(self.modes):
                row = np.zeros(self.dimension )
                col = np.arange(self.dimension , dtype=int)
                data = np.zeros(self.dimension )
                for i in range(self.dimension ):
                    if self.states_list[i][k] == self.local_exc[k] or sum(self.states_list[i]) == self.global_exc:#!!!!!!!!!!!!!!!!!!!!!!
                        row[i] = i 
                    else:
                        current_state = list(self.states_list[i])
                        current_state[k]+=1
                        p = self.index(current_state)
                        row[i] = p
                        data[i]=(self.states_list[i][k]+1)**0.5
                A = coo_matrix((data, (row, col)), shape=(self.dimension , self.dimension ), dtype = complex)
                self.create.append(A)
       
        elif statistics == 'Fermi' or 'Fermi_sea':
        
        #8)
            self.annihilate=[]
            current_state = []
            for k in range(self.modes):
                row = np.zeros(self.dimension )
                col = np.arange(self.dimension , dtype=int)
                data = np.zeros(self.dimension )
                for i in range(self.dimension ):
                    if self.states_list[i][k]==0:
                        row[i] = i
                    else:
                        current_state = list(self.states_list[i])
                        y = sum(current_state[:k])
                        current_state[k] = 0
                        p = self.index(current_state)
                        if (p=='666'):
                            continue
                        row[i] = p
                        data[i]=(-1)**y*(self.states_list[i][k])**0.5
                A = coo_matrix((data, (row, col)), shape=(self.dimension , self.dimension ), dtype = complex)
                self.annihilate.append(A)
                              
         #9)       
            self.create=[]
            current_state = []
            for k in range(self.modes):
                row = np.zeros(self.dimension )
                col = np.arange(self.dimension , dtype=int)
                data = np.zeros(self.dimension )
                for i in range(self.dimension ):
                    if self.states_list[i][k] == self.local_exc[k] or sum(self.states_list[i]) == self.global_exc:
                        row[i] = i 
                    else:
                        current_state = list(self.states_list[i])
                        y = sum(current_state[:k])
                        current_state[k] = 1
                        p = self.index(current_state)
                        if (p=='666'):
                            continue
                        row[i] = p
                        data[i]=(-1)**y*(self.states_list[i][k]+1)**0.5
                A = coo_matrix((data, (row, col)), shape=(self.dimension , self.dimension ), dtype = complex)
                self.create.append(A)
        
        
        
        else:
            print('Сhoose the statistics: Bose, Fermi or Fermi_sea')

             
    
    def sigmax(self, i):

        return(self.annihilate[i] + self.create[i])
    
    
    def sigmay(self, i):
 
        return(1j*(-self.annihilate[i] + self.create[i]))
    
    
    def sigmaz(self, i):

        return(self.annihilate[i]@self.create[i] - self.create[i]@self.annihilate[i])
    
    
    
                
    def ptrace(self, psi, k):
        psi1 = coo_array(psi)
        rho = psi1.conj().T @ psi1
        
        N = self.global_exc
        result = np.zeros((N+1,N+1))
        for p in range(N+1):
            for q in range(N+1):
                N1 = N - max(p,q)
                a = list(self.ngenerator(k, N1, p))
                b = list(self.ngenerator(k, N1, q))
                for i in range(len(a)):
                    result[p,q] += rho[a[i],b[i]]
        return(result)
                            
            
        
      
        
    def occupations(self,i):
     
        if i >= self.dimension :
            print('the number is out of range')
            
        else:
            return(np.array(self.states_list[i]))
    
    
    def index(self, state):
       
        if len(state) != self.modes:
            print('incorrect size of an array')
        else:
            s = tuple(state)
            try:
                return(self.find_index[s])
            except:
                return('666')
    

  
        
    def states_generator(self):
        #Generates all the possible states for given Fock space
        current_state = (0,)*len(self.local_exc1)
        n = 0
        while True:
            yield current_state
            j = len(self.local_exc1) - 1
            current_state = current_state[:j] + (current_state[j]+1,)
            n += 1
            while n > self.global_exc or current_state[j] >= self.local_exc1[j]:
                j -= 1
                if j < 0:
                    return
                n -= current_state[j+1] - 1
                current_state = current_state[:j] + (current_state[j]+1, 0) + current_state[j+2:]
                
    def ngenerator(self, k, N1, q):
        local_exc2 = np.delete(self.local_exc1, k)
        current_state = (0,)*len(local_exc2)
        n = 0
        while True:
            pp = list(current_state)
            pp.insert(k,q)
            pp = tuple(pp)
            
            yield self.find_index[pp]
            
            j = len(local_exc2) - 1
            current_state = current_state[:j] + (current_state[j]+1,)
            n += 1
            while n > N1 or current_state[j] >= local_exc2[j]:
                j -= 1
                if j < 0:
                    return
                n -= current_state[j+1] - 1
                current_state = current_state[:j] + (current_state[j]+1, 0) + current_state[j+2:]
    
    
    def generator_straight(self,m,N):
        current_state = (0,)*m
        n = 0
        while True:
            if (n == N):
                yield current_state
            j = m - 1
            current_state = current_state[:j] + (current_state[j]+1,)
            n += 1
            while n > N or current_state[j] > 1:
                j -= 1
                if j < 0:
                    return
                n -= current_state[j+1] - 1
                current_state = current_state[:j] + (current_state[j]+1, 0) + current_state[j+2:]

                
    def generator_back(self,m,N):
        current_state = (0,)*m
        n = 0
        while True:
            if (n == N):
                yield current_state
            j = 0
            current_state = (current_state[j]+1,) + current_state[j+1:]
            n += 1
            while n > N or current_state[j] > 1:
                j += 1
                if j >= m:
                    return
                n -= current_state[j-1] - 1
                current_state = current_state[:j-1] + (0, current_state[j]+1) + current_state[j+1:]


                
    def sea_generator(self):
        m = self.modes
        m1 = self.fermi_sea_modes
        m2 = m - m1
        N = self.global_exc
        
        n2=0
        h2 = (0,)*m2
        
        if (N>m1):
            for n1 in range (min(m1,N),-1,-1):
                for h1 in self.generator_straight(m1, n1):
                    yield (h1+h2)
            
            
            for n2 in range (1,min(N-m1,m2)+1):
                    for n1 in range (m1,-1,-1):
                        for h2 in self.generator_back(m2, n2):
                            for h1 in self.generator_straight(m1, n1):
                                yield (h1+h2)
            
            if(N-m1<m2):
                for n2 in range (N-m1+1,N-m1+min(m1,m-N)+1):
                    for n1 in range (N-n2,-1,-1):
                        for h2 in self.generator_back(m2, n2):
                            for h1 in self.generator_straight(m1, n1):
                                yield (h1+h2)
        
        else:
            for n1 in range (m1,-1+ m1-N,-1):
                for h1 in self.generator_straight(m1, n1):
                    yield (h1+h2)
                    
            for n2 in range (1,min(m2,N)+1):
                    for n1 in range (N-n2,-1,-1):
                        for h2 in self.generator_back(m2, n2):
                            for h1 in self.generator_straight(m1, n1):
                                yield (h1+h2)
            
                
    



        
class fock_space_kron:
    def __init__(self, f1, f2):

        self.f1 = f1
        self.f2 = f2
        
        #1)
        self.modes = f1.modes + f2.modes

        #2)
        self.dimension  = f1.dimension * f2.dimension

        #3)
        self.emptyH = coo_matrix((self.dimension , self.dimension ), dtype = complex)

        #4)
        self.eye = sparse.eye(self.dimension )
       
        
        if (f1.statistics=='Fermi') and (f2.statistics=='Fermi'):
                       
            data1 = np.zeros(f1.dimension)
            data2 = np.zeros(f2.dimension)
            
            for i in range(f1.dimension):
                k = sum(f1.occupations(i))
                data1[i] = (-1)**k
            for i in range(f2.dimension):
                k = sum(f2.occupations(i))
                data2[i] = (-1)**k
        
        #4.1)
            self.parity_f1 = diags(data1)
        #4.2)
            self.parity_f2 = diags(data2)
        
        #5)
            self.annihilate=[]
        
            for k in range(f1.modes):
                self.annihilate.append(sparse.kron(f1.annihilate[k], f2.eye))
            for k in range(f1.modes, self.modes):
                self.annihilate.append(sparse.kron(self.parity_f1, f2.annihilate[k-f1.modes]))
        #6)
            self.create=[]
        
            for k in range(f1.modes):
                self.create.append(sparse.kron(f1.create[k], f2.eye))
            for k in range(f1.modes, self.modes):
                self.create.append(sparse.kron(self.parity_f1, f2.create[k-f1.modes]))
        
        else:
            
        #5)
            self.annihilate=[]
        
            for k in range(f1.modes):
                self.annihilate.append(sparse.kron(f1.annihilate[k], f2.eye))
            for k in range(f1.modes, self.modes):
                self.annihilate.append(sparse.kron(f1.eye, f2.annihilate[k-f1.modes]))
        #6)
            self.create=[]
        
            for k in range(f1.modes):
                self.create.append(sparse.kron(f1.create[k], f2.eye))
            for k in range(f1.modes, self.modes):
                self.create.append(sparse.kron(f1.eye, f2.create[k-f1.modes]))
                
        self.local_exc = np.concatenate((f1.local_exc,f2.local_exc))
        self.local_exc1 = np.array(self.local_exc+1) #internal parameter
        self.global_exc = f1.global_exc + f2.global_exc
        self.states_list = list(self.states_generator())
        self.find_index = {state: idx for idx, state in enumerate(self.states_list)}
                
                
                
     #ggggggggggg           
                
    def states_generator(self):
        #Generates all the possible states for given Fock space
        current_state = (0,)*len(self.local_exc1)
        n = 0
        m = 0
        j = len(self.local_exc1)
        while True:
            yield current_state

            j = len(self.local_exc1) - 1
            current_state = current_state[:j] + (current_state[j]+1,)
            n += 1
            
            while n > self.f2.global_exc or m > self.f1.global_exc or current_state[j] >= self.local_exc1[j]:
                j -= 1
                if j < 0:
                    return
                if j == (self.f1.modes-1):
                    n -= current_state[j+1]
                    m += 1
                elif j<(self.f1.modes-1):
                    m -= current_state[j+1] - 1
                else:
                    n -= current_state[j+1] - 1
                current_state = current_state[:j] + (current_state[j]+1, 0) + current_state[j+2:]
           
        
    def ngenerator(self, k, N1, N2, q):
        local_exc2 = np.delete(self.local_exc1, k)
        current_state = (0,)*len(local_exc2)
        n = 0
        m = 0
        j = len(self.local_exc1)
        while True:
            pp = list(current_state)
            pp.insert(k,q)
            pp = tuple(pp)
            
            yield self.find_index[pp]
            
            j = len(local_exc2) - 1
            current_state = current_state[:j] + (current_state[j]+1,)
            n += 1
            while n > N1 or m>N2 or current_state[j] >= local_exc2[j]:
                j -= 1
                if j < 0:
                    return
                if j == (self.f1.modes-1):
                    n -= current_state[j+1]
                    m += 1
                elif j<(self.f1.modes-1):
                    m -= current_state[j+1] - 1
                else:
                    n -= current_state[j+1] - 1
                current_state = current_state[:j] + (current_state[j]+1, 0) + current_state[j+2:]
                
    def ptrace(self, psi, k):
        psi1 = coo_array(psi)
        rho = psi1.conj().T @ psi1
        
        N = int(self.local_exc[k])
        N1 = self.f1.global_exc
        N2 = self.f2.global_exc
        border = self.f1.modes
        
        result = np.zeros((N+1,N+1), dtype = complex)
        for p in range(N+1):
            for q in range(N+1):
                if k < border:
                    N1 -= max(p,q)
                else:
                    N2 -= max(p,q)
                a = list(self.ngenerator(k, N1, N2, p))
                b = list(self.ngenerator(k, N1, N2, q))
                for i in range(len(a)):
                    result[p,q] += rho[a[i],b[i]]
        return(result)
                    
                    
                    
                    
    
    def sigmax(self, i):
        if (i<self.f1.modes):
             return(sparse.kron(self.f1.sigmax(i), self.f2.eye))
        else:
            return(sparse.kron(self.f1.eye, self.f2.sigmax(i-self.f1.modes)))

        
    def sigmay(self, i):
       
        if (i<self.f1.modes):
             return(sparse.kron(self.f1.sigmay(i), self.f2.eye))
        else:
            return(sparse.kron(self.f1.eye, self.f2.sigmay(i-self.f1.modes)))

        
    def sigmaz(self, i):
        
        if (i<self.f1.modes):
             return(sparse.kron(self.f1.sigmaz(i), self.f2.eye))
        else:
            return(sparse.kron(self.f1.eye, self.f2.sigmaz(i-self.f1.modes)))
    
    
    def occupations(self,i):
        
        if i >= self.dimension :
            print('the number is out of range')
        else:
            state = np.zeros(self.modes, dtype = int)
            state[:self.f1.modes] = self.f1.occupations(i//self.f2.dimension)
            state[self.f1.modes:] = self.f2.occupations(i%self.f2.dimension)
            return(state)
        
        
    def index(self,state):
        
        if state.size != self.modes:
            print('incorrect size of an array')
        else:
            s = list(state)
            return(self.f1.index(state[:self.f1.modes]) * self.f2.dimension + self.f2.index(state[self.f1.modes:]))
        
        
        
def real_time_solver(psi0, dt, tmax, H, Q = None, final_state = None):
    K = psi0.size
    Nt = int(tmax/dt)+1
    psi = [psi0, psi0]
    
    if callable(H) == False:
        H_const = H
        def H(t):
            return(H_const)
     
    if Q == None:
        for i in range(1,Nt):
            psi_iter_old = psi0
            psi_iter = np.zeros(K)
            psi_compare = psi0
            while (LA.norm(psi_iter-psi_compare)>10**(-6)):
                s = psi[1]+psi_iter_old
                psi_iter = psi[1]+ dt/2*(-1j) * H(i*dt+dt/2)@s
                psi_compare = psi_iter_old
                psi_iter_old = psi_iter
            psi[1] = psi_iter
            
        return (psi[1])
        
    elif type(Q) == list:
        l = len(Q)
        results= np.zeros((l,Nt))
        for j in range(l):
            results[j,0] = abs(np.conj(psi0) @ Q[j] @ psi0)
        
        for i in range(1,Nt):
            psi_iter_old = psi0
            psi_iter = np.zeros(K)
            psi_compare = psi0
            while (LA.norm(psi_iter-psi_compare)>10**(-6)):
                s = psi[1]+psi_iter_old
                psi_iter = psi[1]+ dt/2*(-1j) * H(i*dt+dt/2)@s
                psi_compare = psi_iter_old
                psi_iter_old = psi_iter
            psi[1] = psi_iter
            
            for j in range(l):
                results[j,i] = abs(np.conj(psi[1]) @ Q[j] @ psi[1])
        
    else:
        results = np.zeros(Nt)
        results[0] = abs(np.conj(psi0) @ Q @ psi0)
        
        for i in range(1,Nt):
            psi_iter_old = psi0
            psi_iter = np.zeros(K)
            psi_compare = psi0
            while (LA.norm(psi_iter-psi_compare)>10**(-6)):
                s = psi[1]+psi_iter_old
                psi_iter = psi[1]+ dt/2*(-1j) * H(i*dt+dt/2)@s
                psi_compare = psi_iter_old
                psi_iter_old = psi_iter
            psi[1] = psi_iter
            results[i] = abs(np.conj(psi[1]) @ Q @ psi[1])
    if final_state != None :
        return (results, psi[1])
    else:
        return(results)
    
        
        
        


# In[ ]:




