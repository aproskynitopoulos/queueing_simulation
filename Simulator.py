### Queueing Simulation
#
#   Description:    Generate queueing simulation data for a user-specified network and store the results. When prompted,
#                   enter the desired network and simulation parameters:
#        
#                   Overall thoughput (float):  overall throughput level for the network as a whole. Sets the mean service time 
#                                               at every station to achieve the desired thoughput level (intarrival rates are default as 1)
#                   Interarrival/Service time 
#                   distribution family:        Enter either <Lognormal> or <Exponential> 
#                   COV (float):                If <Lognormal> was entered, this is desired COV for each station
#                   rho (float):                If <Lognormal> was entered, this is desired correlation between the lognormals
#                   Number of runs (int):       The number of simulations that we perform
#                   Number of wait times (int): How many waiting times are simulated at each station during each simulation run

def main():

    from QNetwork import Source, Queue
    import numpy as np
    import heapq
    import pandas as pd
    import csv
    
    base_mean = float(input("Enter overall throughput: "))
    distribution = input("Enter interarrival and service time distribution family: ")
    if distribution == 'Lognormal':
        COV = float(input('Enter CoV: '))
        rho = float(input('Enter rho: '))
    number_of_runs = int(input("Enter number of simulation runs: "))
    number_of_waiting_times = int(input("Enter number of waiting times to be simulated at each station: "))
    
    # specify network structure
    source_matrix = [[1,0,0,0]]
    transition_matrix = [[0,0.5,0.5,0],[0,0,0,1],[0,0,0,1],[0,0,0,0]]
    arrival_means = [1]
    service_means = [0] * len(transition_matrix)
    number_of_sources = len(source_matrix)
    number_of_servers = len(source_matrix[0])
    starts = []

    for i in range(len(source_matrix)):
        for j in range(len(source_matrix[i])):
            if source_matrix[i][j]:
                service_means[j] +=  arrival_means[i] * source_matrix[i][j]
                starts.append((j,arrival_means[i] * source_matrix[i][j]))
    
    while starts:
        loc, power = starts.pop(0)
        for j in range(len(transition_matrix[loc])):
            if transition_matrix[loc][j]:
                service_means[j] +=  power * transition_matrix[loc][j]
                starts.append((j,power * transition_matrix[loc][j]))
    
    service_means = [(1./x) * base_mean for x in service_means]
    number_of_servers = len(source_matrix[0])

    def wrapper(arrival_means, source_matrix,transition_matrix,service_means, final_jobs, distribution, arrival_covs, service_covs, arrival_rho = 0):
        """
        Performs the simulation and stores the final results in a dictionary

        Args:
            arrival_means:  a list containing the mean interarrival times 
            source_matrix:  matrix containing the source probabilities; source_matrix[i][j] is the 
                            probability that a job coming from source i enters the system at node/station j
            service_means:  list of the mean service time at each station
            final_jobs:     the minimum number of departures we wish to observe at every node before the simulation terminates
            distribution:   specifies the interarrival/service time distribution family
            arrival_covs:   list of the COV for each arrival process
            service_covs:   list of the COV for each service process
            arrival_rho:    the correlation between interarrival times from the same outside source
        """
        n = len(transition_matrix[0])
        system = {'Source':Source(arrival_means, source_matrix, distribution, arrival_covs, arrival_rho)}
        
        for i in range(n):
            system[i] = Queue(service_means[i], transition_matrix[i], distribution, service_covs[i])
        
        events = [(system['Source'].time_next,system['Source'].next_job,'Source')]
        heapq.heapify(events)
        Source.time = 0
        Queue.time  = 0
        curr = 0
        jobs = 0
        while jobs < final_jobs:
            curr,job_id,node = heapq.heappop(events)
            Source.time = curr
            Queue.time = curr

            system[node].action_out()
            if system[node].time_next:
                heapq.heappush(events,(system[node].time_next,system[node].next_job,node))
            
            dest = system[node].next_destination
            if dest != n:
                system[dest].action_in(job_id)
                if system[dest].time_next:
                    heapq.heappush(events,(system[dest].time_next,system[dest].next_job,dest))
            jobs = min([len(system[i].departure_times) for i in range(n)])
        
        return system
    
    def extractor(system,k):
        """
        Produces lists of interarrival, service and waiting times for a given station

        Args:   system:     the system (as produced by the wrapper function)
                k:          the index of the station/node for which we want to output the i,s,w times
        """
        res = {}
        if system[k].arrival_times:
           res['interarrival_times'] = [system[k].arrival_times[0]] + [j-i for i, j in zip(system[k].arrival_times[:-1], system[k].arrival_times[1:])]
        else:
            res['interarrival_times'] = []
        if system[k].__class__.__name__== 'Queue' and system[k].service_times:
            res['service_times'] = system[k].service_times
            res['waiting_times'] = [0]
            for i in range(1,len(res['service_times'])):
                new = max(0,res['waiting_times'][-1] + res['service_times'][i-1] - res['interarrival_times'][i])
                res['waiting_times'].append(new)
        else:
            res['service_times'] = []
            res['waiting_times'] = []
        res['job_ids'] = system[k].all_jobs
        
        return res
    
    def export(number_of_runs,number_of_waiting_times):
        """
        Generate the desired number of simulations of our system and store the results

        Args:
            number_of_runs:             the number of times we simulate the system
            number of waiting times:    the number of waiting times we want to have simulated at each station;
                                        once reached at all nodes, the function terminates
        """

        waiting_times = {i:[] for i in range(number_of_servers)}
        job_ids = {i:[] for i in range(number_of_servers)}
        for m in range(number_of_runs):
            print('Simulation number', m+1)
            if distribution == 'Exponential':
                S = wrapper(arrival_means, source_matrix, transition_matrix,service_means, number_of_waiting_times, distribution, [1] * number_of_sources, [1] * number_of_servers)
            if distribution == 'Lognormal':
                S = wrapper(arrival_means, source_matrix, transition_matrix,service_means, number_of_waiting_times, distribution, [COV] * number_of_sources, [COV] * number_of_servers, rho)
            
            for i in range(number_of_servers):
                waiting_times[i].append(extractor(S,i)['waiting_times'][:number_of_waiting_times])
                job_ids[i].append(extractor(S,i)['job_ids'][:number_of_waiting_times])

        # output simulation results as .csv file
        with open('Queue_Network_' + str(number_of_servers) + '_Nodes_' + str(number_of_runs) +'_Runs.csv','w') as csv_file:
           writer = csv.writer(csv_file)
           for i in range(number_of_servers):
            name = 'Server_' + str(i)
            writer.writerow([name])
            for j in range(number_of_runs):
               writer.writerow(['waiting_times_' + str(j)] + waiting_times[i][j])
               writer.writerow(['job_ids_' + str(j)] + job_ids[i][j])

    export(number_of_runs,number_of_waiting_times)

if __name__ == "__main__":
    main()