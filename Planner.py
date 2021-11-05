import numpy as np

class BFS:
        def __init__(self,env):
            self.env=env
            self.drivable_tiles=env.drivable_tiles
            self.map_img, self.goal, self.start_pos = env.get_task_info()
            self.map_data=env.map_data

        def search(self):
            start_node=Node(None,tuple(self.start_pos))
            start_node.g=start_node.h=start_node.f=0
            end_node=Node(None,self.goal)
            tiles = self.map_data['tiles']
            grid_height = len(tiles)
            grid_width = len(tiles[0])
            

            grid=[[0]*grid_height for i in range(grid_width)]

            for i,j in list(map(lambda x:x['coords'],self.drivable_tiles)):
                #print("{},{}".format(i,j))
                grid[i][j]=1


            yet_to_visit_list=[]
            visited_list=[]
            yet_to_visit_list.append(start_node)
            

            while len(yet_to_visit_list)>0:
                current_node=yet_to_visit_list.pop(0)
                visited_list.append(current_node)

                if current_node.position==end_node.position:
                    #print("goal reached")
                    return self.return_plan(current_node)
                
                children=[]

                for node_position, action in self.get_adjacent_action(current_node,grid):
                    new_node=Node(current_node, node_position, action)
                    children.append(new_node)
                
                #print("visting {}: {}".format(current_node.position,list(map(lambda x:x.position,children))))

                for child in children:
                    if len([visited_child for visited_child in visited_list if visited_child==child])>0:
                       continue
                    yet_to_visit_list.append(child)
                
        def return_plan(self,current_node):
            plan={}
            temp=current_node
            while temp.parent is not None:
                plan[str(temp.parent.position)]=temp.position
                temp=temp.parent
            return plan


        def get_adjacent_action(self,node,grid):
            i,j=node.position
            actions=[(1,0),(0,1),(-1,0),(0,-1)]
            adj_pos=[]
            grid_height=len(grid)
            grid_width=len(grid[0])
            for action in actions:
                new_pos=(i+action[0],j+action[1])
                if new_pos[0]>=0 and new_pos[0]<grid_height and new_pos[1]>=0 and new_pos[1]<grid_width and grid[new_pos[0]][new_pos[1]]==1:
                    adj_pos.append((new_pos,action))
            return adj_pos


class Node:
    def __init__(self, parent=None, position=None, action=None):
        self.parent=parent
        self.position=position
        self.action=action
        self.g=0
        self.h=0
        self.f=0
    def __eq__(self, other):
        return self.position==other.position