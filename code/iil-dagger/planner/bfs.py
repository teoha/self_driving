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
        plan={}
        start_node=Node(None,tuple(self.goal))

        yet_to_visit_list=[]
        visited_list=[]
        yet_to_visit_list.append(start_node)
        
        while len(yet_to_visit_list)>0:
            current_node=yet_to_visit_list.pop(0)
            visited_list.append(current_node)

            if current_node.parent is not None:
                plan[current_node.position]=current_node.parent.position

            if current_node.isEdge:
                continue
            
            children=[]

            adj_drivable_pos, adj_nondrivable_pos = self.get_adjacent_action(current_node,self.grid)

            for node_position, action in adj_drivable_pos:
                new_node=Node(current_node, node_position, action)
                children.append(new_node)
            for node_position, action in adj_nondrivable_pos:
                new_node=Node(current_node, node_position, action, True)
                children.append(new_node)

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