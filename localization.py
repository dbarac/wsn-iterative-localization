from pymote.algorithm import NodeAlgorithm
from pymote.message import Message
import math


class IterativeLocalization(NodeAlgorithm):
    """
    Iterative, trilateration-based distributed localization algorithm.

    Assumes that the network is globaly rigid (localizable).
    """
    default_params = {'neighborsKey': 'Neighbors'}

    def initializer(self):
        for node in self.network.nodes():
            # initialize memory and node status
            node.memory[self.neighborsKey] = node.compositeSensor.read()['Neighbors']
            node.memory['neighborDistances'] = node.compositeSensor.read()['Dist']
            node.memory['position'] = None
            node.status = 'WAITING_FOR_FIX'
        ini_node = self.network.nodes()[0] # any node can be the initiator
        ini_node.status = 'INITIATOR'
        self.network.outbox.insert(0, Message(
            header=NodeAlgorithm.INI,
            destination=ini_node
        ))

    def initiator(self, node, message):
        if message.header == NodeAlgorithm.INI:
            node.memory['univisitedNeighbors'] = list(node.memory[self.neighborsKey])
            node.memory['commonNeighborLists'] = {} # key: initiator neighbor node
            # default destination: send to every neighbor
            node.send(Message(
                header='CommonNeighborQuery',
                data=node.memory[self.neighborsKey]
            ))
        elif message.header == 'CommonNeighborResponse':
            node.memory['univisitedNeighbors'].remove(message.source)
            if message.data is not None:
                node.memory['commonNeighborLists'][message.source] = message.data
            if len(node.memory['univisitedNeighbors']) == 0:
                # select two neighbors
                neigh, all_common = node.memory['commonNeighborLists'].popitem() # pick any
                dist_01 = node.memory['neighborDistances'][neigh]
                common_neigh, dist_12 = all_common[0]
                node.memory['commonNeighborLists'].pop(common_neigh)
                dist_02 = node.memory['neighborDistances'][common_neigh]
                print(dist_01, dist_02, dist_12)

                # define initial triangle
                rigid_segment = self.define_initial_rigid_segment(
                    node, dist_01, dist_02, dist_12, neigh, common_neigh
                )
                for neigh, neigh_pos in rigid_segment:
                    node.send(Message(
                        destination=neigh, header='OwnPosition', data=neigh_pos
                    ))

                rigid_nodes = {node for (node,pos) in rigid_segment}
                remaining_neighbors = set(node.memory[self.neighborsKey]) - rigid_nodes
                node.send(Message(
                    header='NeighborPosition', destination=remaining_neighbors,
                    data=node.memory['position']
                ))

                node.status = 'LOCALIZED'

    def waiting_for_fix(self, node, message):
        #if message.header == 'Information':
        #    node.memory[self.informationKey] = message.data
        #    destination_nodes = list(node.memory[self.neighborsKey])
        #    # send to every neighbor-sender
        #    destination_nodes.remove(message.source)
        #    if len(destination_nodes) > 0:
        #        node.send(Message(
        #            destination=destination_nodes,
        #            header='Information',
        #            data=message.data
        #        ))
        if message.header == 'CommonNeighborQuery':
            initiator_neighbors = set(message.data)
            common_neighbors = set(
                node.memory[self.neighborsKey]).intersection(initiator_neighbors)
            # include distance to each common neighbor
            common_neighbors = [
                (n, node.memory['neighborDistances'][n]) for n in common_neighbors
            ]
            if len(common_neighbors) == 0:
                common_neighbors = None
            node.send(Message(
                header='CommonNeighborResponse', destination=message.source,
                data=common_neighbors
            ))
        elif message.header == 'OwnPosition':
            node.memory['position'] = message.data
            # TODO: update (send pos to neighbors, but not to already localized ones
            node.status = 'LOCALIZED'

    def localized(self, node, message):
        pass

    def define_initial_rigid_segment(self, initiator, r_01, r_02, r_12, neigh_1, neigh_2):
        """
        Arbitrarily define an initial rigid segment (triangle) in a way that
        the distances between nodes are the same as measured distances (r_01, r_02, r_12).
        After defining the initial triangle, the rest of the nodes in the
        network can be localized with the trilateration process.
        """
        initiator.memory['position'] = (0,0) # assume initiator position
        x1, y1 = (r_01, 0)
        x2 = (r_01 ** 2 + r_02 ** 2 - r_12 ** 2) / (2 * r_01)
        y2 = math.sqrt(r_02 ** 2 - x2 ** 2)
        rigid_segment = {
            (neigh_1, (x1, y1)), (neigh_2, (x2, y2))
        }
        return rigid_segment

    STATUS = {
        'INITIATOR': initiator,
        'WAITING_FOR_FIX': waiting_for_fix,
        'LOCALIZED': localized,
    }
