### Queueing Network Simulation Classes
#
# Description:      This code creates source and node classes for an open queueing network that is inititally empty. Jobs arrive 
#                   from the outside at Source nodes and are then routed between Queue nodes before they leave the network. 
#                   Interarrival times can be correlated. Every service station in the system operates on FCFS basis and consists of a single queue/server.
#
# Last modified:    October 01, 2023

import numpy as np
import heapq
from collections import deque

class Source:
    
    # the current time in the sytem
    time = 0
    
    def __init__(self, arrival_means, source_matrix, distribution, covs, rho = 0):
        """
        Initialize arrival distributions and arrival nodes

        Args:
            arrival_means:  vector of mean arrival times 
            source_matrix:  matrix specifying outside arrivals; source_matrix[i][j] is the probability that a job from (outside) source i arrivals at node j in the network;
                            each row sums to 1
            distribution:   specify whether interarrival/service processes should have 'Exponential' or 'Lognormal' increments
            covs:           vector of desired coefficients of variations (CoV) at each node/station (only used if distribution == 'Lognormal')
            rho:            correlation between interarrival times (only used if distribution == 'Lognormal')
        """

        self.number_of_sources = len(arrival_means)
        self.number_of_nodes = len(source_matrix[0])
        self.source_matrix = source_matrix
        self.distribution = distribution
        
        # specifying interarrival distributions
        if distribution == 'Exponential':
            self.mu = arrival_means
            self.sigma = [x ** 2 for x in self.mu]
            self.next_arrival = [(np.random.exponential(self.mu[i]),i) for i in range(self.number_of_sources)]
        
        if distribution == 'Lognormal':
            self.rho = rho
            self.sigma = [np.sqrt(np.log(1 + x ** 2)) for x in covs]
            self.mu = [np.log(arrival_means[i]) - (self.sigma[i] ** 2)/2 for i in range(self.number_of_sources)]
            self.next_arrival = [(np.random.lognormal(self.mu[i],self.sigma[i]),i) for i in range(self.number_of_sources)]
            # the normals that correspond to the simulated lognormals
            self.previous_arrival = [(np.log(self.next_arrival[i][0]) -  self.mu[i])/self.sigma[i] for i in range(self.number_of_sources)] 
        
        # initializing source job heap
        heapq.heapify(self.next_arrival)
        self.time_next = self.next_arrival[0][0]
        self.next_source = self.next_arrival[0][1]
        self.next_job = 0
        self.next_destination = np.random.choice(range(self.number_of_nodes), p = self.source_matrix[self.next_source])
    
    def action_out(self):
        """ 
        Generates the stream of jobs entering the system from the outside source node(s) by maintaining a heap
        """

        # create next arrival time
        if self.distribution == 'Lognormal':
            
            last_normal_time = self.previous_arrival[self.next_source]
            dummy_normal = np.random.normal(loc = 0.0, scale = 1.0, size = None)
            
            s = self.sigma[self.next_source]
            m = self.mu[self.next_source]

            rho_normal = np.log(self.rho * (np.exp(s ** 2) - 1) + 1) / (s ** 2)
            new_normal = rho_normal * last_normal_time + np.sqrt(1 - rho_normal ** 2) * dummy_normal
            new_lognormal = np.exp(m + new_normal * s)
            
            new_time = self.time + new_lognormal
            self.previous_arrival[self.next_source] = new_normal
        
        if self.distribution == 'Exponential':
            new_time = self.time + np.random.exponential(self.mu[self.next_source])
        
        # update arrival stream
        heapq.heappop(self.next_arrival)
        heapq.heappush(self.next_arrival,(new_time,self.next_source))
        self.time_next = self.next_arrival[0][0]
        self.next_source = self.next_arrival[0][1]
        self.next_job += 1
        self.next_destination = np.random.choice(range(self.number_of_nodes), p = self.source_matrix[self.next_source])

class Queue:
    
    # the current time in the sytem
    time = 0
    
    def __init__(self,service_mean,destination,distribution,cov):
        """
        Creates a queueing station within the network

        Args:
            service_mean:   the mean service time at the station
            destination:    list containing transition probabilities; destination[j] is the probability that a job leaving the current node arrives at node j next
            distribution:   the service distribution; either 'Exponential' or 'Lognormal'
            cov:            the coefficient of variation of the service time distribution at the station
        """
        self.p_out = 1 - sum(destination)
        self.destination = destination
        self.number_of_nodes = len(destination)
        self.distribution = distribution

        # generate service time distribution
        if distribution == 'Exponential':
            self.mu = service_mean
            self.sigma = self.mu ** 2

        if distribution == 'Lognormal':
            self.sigma = np.sqrt(np.log(1 + cov ** 2))
            self.mu = np.log(service_mean) - (self.sigma ** 2)/2
        
        # initialize empty queue as well as service, arrival and departure time histories at that node
        self.in_system = 0
        self.all_jobs = []
        self.jobs_in_system = deque([])
        self.next_job = None
        self.time_next = 0
        self.arrival_times = []
        self.service_times = []
        self.departure_times = []
        self.finish_time = 0
        self.next_destination = None
    
    def action_in(self,job_id):
        """
        Maintains the station's queue upon entrance of a job from another node (or outside source); generates its service time if it is next in line
        
        Args:
            job_id:     id of entering job
        """
        
        self.in_system += 1
        self.all_jobs.append(job_id)
        self.jobs_in_system.append(job_id)
        self.arrival_times.append(self.time)
        if self.in_system > 1:
            ######
            self.time_next = None
        else:
            if self.distribution == 'Lognormal':
                service_new = np.random.lognormal(self.mu,self.sigma)
            if self.distribution == 'Exponential':
                service_new = np.random.exponential(self.mu)
            
            self.next_job = self.jobs_in_system.popleft()
            self.service_times.append(service_new)
            self.finish_time = self.time + service_new
            self.time_next = self.finish_time
    
    def action_out(self):
        """
        Maintains the station's queue upon departure of a job to another node (or from the system)

        Args: 
        """
        
        self.in_system -= 1
        if self.in_system:
            if self.distribution == 'Lognormal':
                service_new = np.random.lognormal(self.mu,self.sigma)
            if self.distribution == 'Exponential':
                service_new = np.random.exponential(self.mu)

            self.next_job = self.jobs_in_system.popleft()
            self.finish_time = self.time + service_new
            self.service_times.append(service_new)
            self.time_next = self.finish_time
        else:
            self.time_next = None
        self.departure_times.append(self.time)
        self.next_destination = np.random.choice(range(self.number_of_nodes+1), p = self.destination + [self.p_out])
