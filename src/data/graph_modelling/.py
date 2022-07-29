

    def closure_node_inheritance(self,node):
        closure_node = nx.MultiDiGraph()
        superclass_subclass = {}

        tot_classes = []

        if node[0:4].startswith("http"): #non è una proprietà
            superclasses = self.get_superclass(node)
            tot_classes.append(node)
            for superclass in superclasses:
                tot_classes.append(superclass)

                if superclass not in list(superclass_subclass.keys()):
                    superclass_subclass[superclass] = []
                superclass_subclass[superclass].append(node) 
        
        for node in tot_classes:
            node = str(node)
            query = " SELECT DISTINCT ?property ?class WHERE {"+\
            " ?property rdfs:domain <"+ node +">."+\
            " ?property rdfs:range  ?class . ?class a " + self.config["ontology"]["class"]+".} "
            
            result = self.ontology.query(query)
            for r in result:
                rel = str(r[0])
                obj = str(r[1])

                if node in list(superclass_subclass.keys()) and obj in list(superclass_subclass.keys()):
                    weight = 10
                    for subclass_obj in superclass_subclass[obj]:
                        for subclass_node in superclass_subclass[node]:
                            if not self.exists_edge(closure_node, subclass_obj, subclass_node, rel):
                                closure_node.add_edge(subclass_obj, subclass_node, label = rel, weight = weight)
                elif node in list(superclass_subclass.keys()) and obj not in list(superclass_subclass.keys()):
                    weight = 10
                    for subclass_node in superclass_subclass[node]:
                        if not self.exists_edge(closure_node,subclass_node, obj, rel):
                            closure_node.add_edge(subclass_node, obj, label = rel, weight = weight)
                elif node not in list(superclass_subclass.keys()) and obj in list(superclass_subclass.keys()):
                    weight = 10
                    for subclass_obj in superclass_subclass[obj]:
                        if not self.exists_edge(closure_node, node, subclass_obj, rel):
                            closure_node.add_edge(node, subclass_obj, label = rel, weight = weight)
                else:
                    weight = 1
                    if not self.exists_edge(closure_node, node, obj, rel):
                        closure_node.add_edge(node, obj, label = rel, weight = weight)

            query = " SELECT DISTINCT ?property ?class WHERE {"+\
            " ?property rdfs:range <"+ node +">."+\
            " ?property rdfs:domain  ?class . ?class a " + self.config["ontology"]["class"]+".} "
            
            result = self.ontology.query(query)
            for r in result:
                rel = str(r[0])
                subj = str(r[1])

                if node in list(superclass_subclass.keys()) and subj in list(superclass_subclass.keys()):
                    weight = 10
                    for subclass_subj in superclass_subclass[subj]:
                        for subclass_node in superclass_subclass[node]:
                            if not self.exists_edge(closure_node, subclass_subj, subclass_node, rel):
                                closure_node.add_edge(subclass_subj, subclass_node, label = rel, weight = weight)
                elif node in list(superclass_subclass.keys()) and subj not in list(superclass_subclass.keys()):
                    weight = 10
                    for subclass_node in superclass_subclass[node]:
                        if not self.exists_edge(closure_node, subj, subclass_node, rel):
                            closure_node.add_edge(subj, subclass_node, label = rel, weight = weight)
                elif node not in list(superclass_subclass.keys()) and subj in list(superclass_subclass.keys()):
                    weight = 10
                    for subclass_subj in superclass_subclass[subj]:
                        if not self.exists_edge(closure_node, subclass_subj, node, rel):
                            closure_node.add_edge(subclass_subj, node, label = rel, weight = weight)
                else:
                    weight = 1
                    if not self.exists_edge(closure_node, subj, node, rel):
                        closure_node.add_edge(subj, node, label = rel, weight = weight)
        return closure_node





    def semantic_description(self,semantic_model):
        #closure = self.compute_closure_node("http://dbpedia.org/ontology/Director")
        #return closure
        Uc_occurrences = {}

        Uc = [] 
        Ut = []
        Et = []
        Er = []
        Uc_ini = []
        #init UC and Ut
        for node in semantic_model.nodes:
            if node[0:4].startswith("http"):
                Uc.append(node)
                Uc_ini.append(node)
                Uc_occurrences[node[0:len(node)-1]] = Uc_occurrences.get(node[0:len(node)-1],0)+1
            else:
                Ut.append(node)

        #Init Et and Er
        for edge in semantic_model.edges:
            label = semantic_model.get_edge_data(edge[0], edge[1])[0]
            if edge[0][0:4].startswith("http") and edge[1][0:4].startswith("http"):
                Er.append(label["label"])
            else:
                
                Et.append(label["label"])

                #print(edge[0],edge[1], closure_C.get_edge_data(edge[0],edge[1]))
        closure_classes = []
        
        for uc in Uc_ini:
            C = uc[0: len(uc)-1]
            if C not in closure_classes:
                closure_classes.append(C)
            closure_C = self.compute_closure_node(C)
            for edge in closure_C.out_edges:
                if len(self.get_superclass(edge[0])) != 0 and not self.class_exists_instances(edge[0], Uc_ini):
                    continue
                if len(self.get_superclass(edge[1])) != 0 and not self.class_exists_instances(edge[1], Uc_ini):
                    continue
                if edge[0] not in closure_classes:
                    closure_classes.append(edge[0])
                if edge[1] not in closure_classes:
                    closure_classes.append(edge[1])

        for uc in Uc_ini:
            us = ""
            C = uc[0: len(uc)-1]

            closure_C = self.closure_node_inheritance(C)
            #self.draw_result(closure_C, "/home/sara/Desktop/fase2/git_repo/knowledge-graph-learning/data/graph_images/closure_node111")
            
            for edge in closure_C.out_edges:
                #print(edge[0],edge[1], closure_C.get_edge_data(edge[0],edge[1]))
                epsilon = 10
                C1 = edge[0]
                C2 = edge[1]
                relations=[]
                rel = closure_C.get_edge_data(C1,C2)
                for i in range(len(rel)):
                    relations.append(rel[i]["label"])

                us_list =[]
                ut_list =[]
                if self.is_subclass(C, C1) or C==C1:
                    us_list.append(uc)
                else:
                    uc1 = C1+"0"
                    if uc1 not in Uc:
                        if not self.is_superclass_or_subclass_of(uc1, Uc):
                            if len(self.get_superclass(C1)) == 0:
                                if C1 not in closure_classes:
                                    closure_classes.append(C1)
                                if C1 not in Uc_occurrences:
                                    Uc_occurrences[C1] = 1
                                us_list.append(uc1)
                                Uc.append(uc1)
                            elif len(self.get_superclass(C1)) != 0 and C1 in Uc:
                                us_list.append(uc1)
                                Uc.append(uc1)

                        else:
                            subclasses = self.get_subclasses(C1)
                             #superclasses = self.get_superclass(C1)
                            if len(subclasses)!= 0:
                                for subclass in subclasses:
                                    k = Uc_occurrences.get(subclass,0)
                                    for i in range(k):
                                        us = subclass+str(i)
                                        if subclass in closure_classes:
                                            us_list.append(us)
                                    if k == 0 and subclass in closure_classes:
                                        us_list.append(subclass+"0")
                    else:
                        us_list.append(uc1)

                if self.is_subclass(C, C2) or C == C2:
                    ut_list.append(uc)
                else:
                    uc2 = C2+"0"
                    if uc2 not in Uc:
                        
                        if not self.is_superclass_or_subclass_of(uc2, Uc):
                            Uc.append(uc2)

                            if C2 not in closure_classes:
                                closure_classes.append(C2)
                            if C2 not in Uc_occurrences:
                                Uc_occurrences[C2] = 1
                            ut_list.append(uc2)

                            superclasses = self.get_superclass(C2)
                            if len(superclasses) != 0:
                                for superclass in superclasses:
                                    subclasses = self.get_subclasses(superclass)
                                    for subclass in subclasses:
                                        if subclass != C2 and subclass in closure_classes:
                                            ut_list.append(subclass+"0")
                        else:
                            subclasses = self.get_subclasses(C2)
                            if len(subclasses)!= 0:
                                for subclass in subclasses:
                                    k = Uc_occurrences.get(subclass,0)
                                    for i in range(k):
                                        ut = subclass+str(i)
                                        if subclass in closure_classes:
                                            ut_list.append(ut)
                                    if k == 0 and subclass in closure_classes:
                                        ut_list.append(subclass+"0")
                    else:
                        ut_list.append(uc2)

                if len(us_list) == 0 or len(ut_list) == 0:
                    continue

                us_list, ut_list = self.homogenize_lists(us_list, ut_list)

                for r in relations:
                    for i in range(len(us_list)):
                        us = us_list[i]
                        for j in range(len(us_list)):
                            ut = ut_list[j]

                            H = Uc_occurrences.get(us[0:len(us)-1],0)
                            K = Uc_occurrences.get(ut[0:len(ut)-1],0)
                            h = int(us[len(us)-1:])
                            k = int(ut[len(ut)-1:])

                            Pr_source = self.get_distance(C1,us[0:len(us)-1])
                            Pr_dest = self.get_distance(C2,ut[0:len(ut)-1])
                            Pr = (Pr_source + Pr_dest)*epsilon

                            if h != k:
                                Pr += 10
                            if us != ut and (us, r, ut, Pr) not in Er and (ut, r, us, Pr) not in Er:
                                if ((us[0:len(us)-1] == ut[0:len(ut)-1]) or ( h == k) 
                                    or (H <= K and h == H-1 and k > h) or (K-1 == k and h > k)):

                                    if self.check_relation_exists(us,r,ut):
                                        Er.append((us,r,ut, Pr))
        return (Uc, Er)
