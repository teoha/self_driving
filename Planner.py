import numpy as np

class BFS:
        def __init__(self,env,grid):
            self.grid=grid
            self.map_img, self.goal, self.start_pos = env.get_task_info()

        def search(self):
            plan={}
            start_node=Node(None,tuple(self.goal))
            # end_node=Node(None,self.goal)

            yet_to_visit_list=[]
            visited_list=[]
            yet_to_visit_list.append(start_node)
            
            #print(self.grid)

            while len(yet_to_visit_list)>0:
                current_node=yet_to_visit_list.pop(0)
                visited_list.append(current_node)

                if current_node.parent is not None:
                    plan[str(current_node.position)]=current_node.parent.position

                if current_node.isEdge:
                    continue
                
                children=[]

                adj_drivable_pos, adj_nondrivable_pos = self.get_adjacent_action(current_node,self.grid)

                #print(adj_drivable_pos)

                for node_position, action in adj_drivable_pos:
                    new_node=Node(current_node, node_position, action)
                    children.append(new_node)
                for node_position, action in adj_nondrivable_pos:
                    new_node=Node(current_node, node_position, action, True)
                    children.append(new_node)
                #print("visting {}: {}".format(current_node.position,list(map(lambda x:x.position,children))))

                for child in children:
                    if len([visited_child for visited_child in visited_list if visited_child==child])>0:
                       continue
                    yet_to_visit_list.append(child)
            return plan


        def get_adjacent_action(self,node,grid):
            i,j=node.position
            actions=[(1,0),(0,1),(-1,0),(0,-1)]
            adj_drivable_pos=[]
            adj_nondrivable_pos=[]
            grid_height=len(grid)
            grid_width=len(grid[0])
            for action in actions:
                new_pos=(i+action[0],j+action[1])
                #print(new_pos)
                #print(grid[new_pos[0]][new_pos[1]])
                if new_pos[1]>=0 and new_pos[1]<grid_height and new_pos[0]>=0 and new_pos[0]<grid_width and grid[new_pos[1]][new_pos[0]]:
                    adj_drivable_pos.append((new_pos,action))
                else:
                    adj_nondrivable_pos.append((new_pos,action))
            return adj_drivable_pos, adj_nondrivable_pos


class Node:
    def __init__(self, parent=None, position=None, action=None, isEdge=False):
        self.parent=parent
        self.position=position
        self.action=action
        self.isEdge=isEdge

    def __eq__(self, other):
        return self.position==other.position