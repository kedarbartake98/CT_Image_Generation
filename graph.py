def get_graph():

	n = 12
	gap = 24

	'''Creating static graph structure for all instances'''
	patient_seg_adj_dict = {}
	for i in range(n + gap):
	    patient_seg_adj_dict[i] = []

	for i in range(n):
	    for j in range(n):
	        if i!=j:
	            patient_seg_adj_dict[i].append(j)

	add_co = n
	for i in range(n):
	    patient_seg_adj_dict[i].append(add_co)
	    patient_seg_adj_dict[add_co].append(i)
	    patient_seg_adj_dict[add_co].append(add_co + 1)
	    patient_seg_adj_dict[add_co + 1].append(add_co)
	    add_co+=1
	    if i+1 != n:
	        patient_seg_adj_dict[i+1].append(add_co)
	        patient_seg_adj_dict[add_co].append(i+1)
	        add_co+=1
	    else:
	        patient_seg_adj_dict[0].append(add_co)
	        patient_seg_adj_dict[add_co].append(0)

	# print(patient_seg_adj_dict)
	ll = []
	for x in range(12):
	    ll.append([x, x*2 + 12, x*2 + 13, x+1])
	# ll.append([11,0])
	ll[-1][-1] = 0
	# ll
	co_in = []


	for i,z in enumerate(ll):
	    if i == 0:
	        co_in.extend(z)
	    else:
	        co_in.extend(z[1:])

	#########################################################
	n = 6
	gap = 12

	'''Creating static graph structure for all instances'''
	patient_seg_adj_dict123 = {}
	for i in range(n + gap):
	    patient_seg_adj_dict123[i] = []

	for i in range(n):
	    for j in range(n):
	        if i!=j:
	            patient_seg_adj_dict123[i].append(j)

	add_co = n
	for i in range(n):
	    patient_seg_adj_dict123[i].append(add_co)
	    patient_seg_adj_dict123[add_co].append(i)
	    patient_seg_adj_dict123[add_co].append(add_co + 1)
	    patient_seg_adj_dict123[add_co + 1].append(add_co)
	    add_co+=1
	    if i+1 != n:
	        patient_seg_adj_dict123[i+1].append(add_co)
	        patient_seg_adj_dict123[add_co].append(i+1)
	        add_co+=1
	    else:
	        patient_seg_adj_dict123[0].append(add_co)
	        patient_seg_adj_dict123[add_co].append(0)

	# print(patient_seg_adj_dict)
	ll123 = []
	for x in range(6):
	    ll123.append([x, x*2 + 6, x*2 + 7, x+1])
	# ll123.append([11,0])
	ll123[-1][-1] = 0
	# ll123
	co_in123 = []


	for i,z in enumerate(ll123):
	    if i == 0:
	        co_in123.extend(z)
	    else:
	        co_in123.extend(z[1:])
	
	######################################################
	n = 3
	gap = 6

	'''Creating static graph structure for all instances'''
	patient_seg_adj_dict45 = {}
	for i in range(n + gap):
	    patient_seg_adj_dict45[i] = []

	for i in range(n):
	    for j in range(n):
	        if i!=j:
	            patient_seg_adj_dict45[i].append(j)

	add_co = n
	for i in range(n):
	    patient_seg_adj_dict45[i].append(add_co)
	    patient_seg_adj_dict45[add_co].append(i)
	    patient_seg_adj_dict45[add_co].append(add_co + 1)
	    patient_seg_adj_dict45[add_co + 1].append(add_co)
	    add_co+=1
	    if i+1 != n:
	        patient_seg_adj_dict45[i+1].append(add_co)
	        patient_seg_adj_dict45[add_co].append(i+1)
	        add_co+=1
	    else:
	        patient_seg_adj_dict45[0].append(add_co)
	        patient_seg_adj_dict45[add_co].append(0)

	# print(patient_seg_adj_dict)
	ll145 = []
	for x in range(3):
	    ll145.append([x, x*2 + 3, x*2 + 4, x+1])
	# ll145.append([11,0])
	ll145[-1][-1] = 0
	# ll145
	co_in45 = []


	for i,z in enumerate(ll145):
	    if i == 0:
	        co_in45.extend(z)
	    else:
	        co_in45.extend(z[1:])


	return co_in, co_in123, co_in45
