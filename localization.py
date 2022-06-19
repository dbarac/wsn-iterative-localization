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
            node.memory['neighborPositions'] = {}
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
                localized_neighbors = {node for (node,pos) in rigid_segment}
                for neigh, neigh_pos in rigid_segment:
                    node.send(Message(
                        destination=neigh, header='OwnPosition',
                        data=(neigh_pos, localized_neighbors)
                    ))

                remaining_neighbors = set(node.memory[self.neighborsKey]) - localized_neighbors
                node.send(Message(
                    header='NeighborPosition', destination=remaining_neighbors,
                    data=node.memory['position']
                ))

                node.status = 'LOCALIZED'

    def waiting_for_fix(self, node, message):
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
            pos, localized_nodes = message.data
            node.memory['position'] = pos
            localized_nodes.add(message.source)
            remaining_neighbors = set(node.memory[self.neighborsKey]) - localized_nodes
            node.send(Message(
                header='NeighborPosition', destination=remaining_neighbors,
                data=node.memory['position']
            ))
            node.status = 'LOCALIZED'
        elif message.header == 'NeighborPosition':
            node.memory['neighborPositions'][message.source] = message.data
            if len(node.memory['neighborPositions']) == 3:
                n1, n2, n3 = node.memory['neighborPositions'].keys()
                (x1, y1), (x2, y2), (x3, y3) = node.memory['neighborPositions'].values()
                r1, r2, r3 = (
                    node.memory['neighborDistances'][neigh] for neigh in (n1, n2, n3)
                )
                node.memory['position'] = trilaterate(x1, y1, x2, y2, x3, y3, r1, r2, r3)
                nonlocalized_neighbors = set(node.memory[self.neighborsKey]) - {n1, n2, n3}
                node.send(Message(
                    header='NeighborPosition', destination=nonlocalized_neighbors,
                    data=node.memory['position']
                ))
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
        initiator.memory['position'] = (0, 0) # assume initiator position
        x1, y1 = (r_01, 0)
        x2 = (r_01 ** 2 + r_02 ** 2 - r_12 ** 2) / (2 * r_01)
        y2 = math.sqrt(r_02 ** 2 - x2 ** 2)
        rigid_segment = {
            (neigh_1, (x1, y1)), (neigh_2, (x2, y2))
        }
        return rigid_segment

    def trilaterate(self, x1, y1, x2, y2, x3, y3, r1, r2, r3):
        A = -2 * x1 + 2 * x2
        B = -2 * y1 + 2 * y2
        C = r1 ** 2 - r2 ** 2 - x1 ** 2 + x2 ** 2 - y1 ** 2 + y2 ** 2
        D = -2 * x2 + 2 * x3
        E = -2 * y2 + 2 * y3
        F = r2 ** 2 - r3 ** 2 - x2 ** 2 + x3 ** 2 - y2 ** 2 + y3 ** 2
        x = (C * E - F * B) / (E * A - B * F)
        y = (C -A * x) / B
        return (x, y)

    STATUS = {
        'INITIATOR': initiator,
        'WAITING_FOR_FIX': waiting_for_fix,
        'LOCALIZED': localized,
    }
