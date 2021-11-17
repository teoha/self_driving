from math import *
class BFS:
    """
    Planner to output search path

    ...

    Attributes
    ----------
    goal : int, int
        coordinates of goal tile
    start_pos : int, int
        coordinates of start position
    grid : list[list[bool]]
        grid indicating drivable tiles

    Methods
    -------
    search():
        returns search path as a dict
    
    """

    def __init__(self, goal, start_pos, grid):
        self.grid = grid
        self.goal = goal
        self.start_pos = start_pos

    def search(self):
        """
        Returns search path as a dictionary.
        Each key, value pair indicates cur_tile, next_tile.

        Returns
        -------
        path : dict
            dictionary indicating next_tile for each cur_tile
        """

        # print("GOAL:{}".format(self.goal))
        start_node=Node(None,(self.start_pos[1],self.start_pos[0],self.start_pos[2]))
        start_node.g=start_node.h=start_node.f=0
        end_node=Node(None, (self.goal[0],self.goal[1], 0))

        yet_to_visit_list=[]
        visited_list=[]
        yet_to_visit_list.append(start_node)

        while len(yet_to_visit_list)>0:
            # input()
            current_node=yet_to_visit_list[0]
            current_index=0
            for index,item in enumerate(yet_to_visit_list):
                if item.f<current_node.f:
                    current_node=item
                    current_index=index
            yet_to_visit_list.pop(current_index)
            visited_list.append(current_node)
            if current_node.position[0]==self.goal[0] and current_node.position[1]==self.goal[1]:
                # print("GOAL REACHED: {}".format(current_node.position))
                return self.return_plan(current_node)
            
            # print("curr: {}".format(current_node.position))
            children=[]

            adj_drivable_pos = self.get_adjacent_action(current_node,self.grid)

            for node_position, action in adj_drivable_pos:
                # print(node_position)
                new_node=Node(current_node, node_position, action)
                children.append(new_node)

            for child in children:
                if len([visited_child for visited_child in visited_list if visited_child==child])>0:
                    continue
                child.g=current_node.g+1
                child.h=self._d_from_goal(child.position)
                child.f=child.g+child.h
                
                if len([i for i in yet_to_visit_list if child==i and child.g>i.g])>0:
                    continue
                yet_to_visit_list.append(child)

    def _d_from_goal(self, pose):
     return sqrt((pose[0] - self.goal[0])**2 + (pose[1] - self.goal[1])**2)
    
    def return_plan(self,current_node):
        plan={}
        temp=current_node
        while temp.parent is not None:
            # plan.insert(0,temp.action)
            plan[temp.parent.position]=temp.action
            temp=temp.parent
        return plan




    def get_adjacent_action(self,node,grid):
        x,y,theta=node.position
        actions=[(1,0),(0,-1),(0,1)]
        adj_drivable_pos=[]
        grid_height=len(grid)
        grid_width=len(grid[0])
        #print(node.position)
        for action in actions:
            new_orientation=(theta-action[1])%4
            if theta==0: #East
                new_pose = x+action[0],y+action[1]
            elif theta==1: #North
                new_pose = x+action[1],y-action[0]
            elif theta==2: #West
                new_pose = x-action[0], y-action[1]
            else: #South
                new_pose = x-action[1], y+action[0]

            new_pose=(*new_pose,new_orientation)
            if new_pose[1]>=0 and new_pose[1]<grid_height and new_pose[0]>=0 and new_pose[0]<grid_width and grid[new_pose[1]][new_pose[0]]:
                # print(new_pose)
                if action==(1,0):
                    action_str="FORWARD"
                elif action==(0,-1):
                    action_str="LEFT"
                else:
                    action_str="RIGHT"
                adj_drivable_pos.append((new_pose,action))
        # print("===============")
        return adj_drivable_pos


class Node:
    def __init__(self, parent=None, position=None, action=None, isEdge=False):
        self.parent=parent
        self.position=position
        self.action=action
        self.g=0
        self.h=0
        self.f=0

    def __eq__(self, other):
        return self.position==other.position