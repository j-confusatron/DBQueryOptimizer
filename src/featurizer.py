NODE_TYPES = {
    "Nested Loop":          0,
    "Hash Join":            1,
    "Merge Join":           2,
    "Seq Scan":             3,
    "Index Scan":           4,
    "Index Only Scan":      5,
    "Bitmap Index Scan":    6,
    "Other":                7
}

class Node():
    def __init__(self, buffers=None) -> None:
        self.ntype = 0
        self.ntype_id = 0
        self.cost = 0
        self.rows = 0
        self.bf_pkey = 0
        self.bf_n = 0
        self.is_root = 0
        self.gen = 0
        self.descendants = 0
        self.has_l_child = 0
        self.has_r_child = 0
        self.r_ntype = 0
        self.r_ntype_id = 0
        self.r_cost = 0
        self.r_rows = 0

        self.relation = None
        self.prev = None
        self.next = None
        self.buffers = buffers

    def featurize(self):
        return [
            self.ntype,
            self.ntype_id,
            self.cost,
            self.rows,
            self.bf_pkey,
            self.bf_n,
            self.is_root,
            self.gen,
            self.descendants,
            self.has_l_child,
            self.has_r_child,
            self.r_ntype,
            self.r_ntype_id,
            self.r_cost,
            self.r_rows
        ]

    def to_list(self):
        l = [self.featurize()]
        n = self.next
        while n:
            l.append(n.featurize())
            n = n.next
        return l

    def build(self, node, parent=None):
        self.ntype = NODE_TYPES[node['Node Type']]
        self.ntype_id = int(node['Node Type ID'])
        self.cost = node['Total Cost']
        self.rows = node['Plan Rows']
        self.relation = node['Relation Name'] if 'Relation Name' in node else None

        if parent:
            self.gen = parent.gen + 1
            self.prev = parent
        else:
            self.is_root = 1

        children = []
        if('Plans' in node):
            for n in node['Plans']:
                children.append(Node(buffers=self.buffers).build(n, self))

        if len(children) == 1:
            if self.ntype == NODE_TYPES['Other'] and not children[0].next:
                self.ntype = children[0].ntype
                self.ntype_id = children[0].ntype_id
                if children[0].relation and not self.relation:
                    self.relation = children[0].relation
            else:
                self.has_l_child = 1
                self.descendants = children[0].descendants + 1
                self.next = children[0]

        elif len(children) == 2:
            self.has_l_child = 1
            self.has_r_child = 1
            if children[0].next:
                self.descendants = children[0].descendants + 1
                self.next = children[0]
                self.r_ntype = children[1].r_ntype
                self.r_ntype_id = children[1].r_ntype_id
                self.r_cost = children[1].r_cost
                self.r_rows = children[1].r_rows
            else:
                self.descendants = children[1].descendants + 1
                self.next = children[1]
                self.r_ntype = children[0].ntype
                self.r_ntype_id = children[0].ntype_id
                self.r_cost = children[0].cost
                self.r_rows = children[0].rows

        if self.buffers and self.relation and self.relation in self.buffers:
            bf = self.buffers[self.relation]
            self.bf_pkey = bf['pkey'] if 'pkey' in bf else 0
            self.bf_n = bf['n'] if 'n' in bf else 0

        return self

def buffer_to_dict(buffer):
    d_buffer = {}
    for k, v in buffer.items():
        if (i_pkey := k.find('_pkey')) > 0:
            t = k[:i_pkey]
            if t not in d_buffer:
                d_buffer[t] = {'pkey': v}
            else:
                d_buffer[t]['pkey'] = v
        elif (i_id := k.find('_id_')) > 0:
            t = k[i_id+4:]
            id = k[:i_id]
            if t not in d_buffer:
                d_buffer[t] = {'id': {id: v}}
            elif 'id' not in d_buffer[t]:
                d_buffer[t]['id'] = {id: v}
            else:
                d_buffer[t]['id'][id] = v
        else:
            if k not in d_buffer:
                d_buffer[k] = {'n': v}
            else:
                d_buffer[k]['n'] = v
    return d_buffer

def featurize(plans, buffers):
    # Build out each of the plans as a list.
    buffers = buffer_to_dict(buffers)
    node_plans = [Node(buffers=buffers).build(plan['Plan']).to_list() for plan in plans]

    # Find the max plan length.
    max_plan = 0
    for plan in node_plans:
        l_plan = len(plan)
        if l_plan > max_plan:
            max_plan = l_plan

    # Compress all plans into a single list.
    features = [[] for _ in range(max_plan)]
    for i in range(max_plan):
        for plan in node_plans:
            features[i] += plan[i] if i < len(plan) else Node().featurize()

    return features



# Debug
if __name__ == '__main__':
    plans = [{'Plan': {'Node Type': 'Other', 'Node Type ID': '42', 'Total Cost': 19532.637021, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Other', 'Node Type ID': '45', 'Total Cost': 19532.612021, 'Plan Rows': 2.0, 'Plans': [{'Node Type': 'Other', 'Node Type ID': '42', 'Total Cost': 18532.412021, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Nested Loop', 'Node Type ID': '36', 'Total Cost': 18532.162021, 'Plan Rows': 32.0, 'Plans': [{'Node Type': 'Hash Join', 'Node Type ID': '38', 'Total Cost': 18513.45993, 'Plan Rows': 32.0, 'Plans': [{'Node Type': 'Nested Loop', 'Node Type ID': '36', 'Total Cost': 18511.68643, 'Plan Rows': 131.0, 'Plans': [{'Node Type': 'Hash Join', 'Node Type ID': '38', 'Total Cost': 15253.600333, 'Plan Rows': 5089.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'movie_info_idx', 'Total Cost': 13685.145833, 'Plan Rows': 575015.0}, {'Node Type': 'Other', 'Node Type ID': '47', 'Total Cost': 2.4125, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'info_type', 'Total Cost': 2.4125, 'Plan Rows': 1.0}]}]}, {'Node Type': 'Index Scan', 'Node Type ID': '21', 'Relation Name': 'movie_companies', 'Total Cost': 0.630221, 'Plan Rows': 1.0}]}, {'Node Type': 'Other', 'Node Type ID': '47', 'Total Cost': 1.05, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'company_type', 'Total Cost': 1.05, 'Plan Rows': 1.0}]}]}, {'Node Type': 'Index Scan', 'Node Type ID': '21', 'Relation Name': 'title', 'Total Cost': 0.57194, 'Plan Rows': 1.0}]}]}]}]}}, {'Plan': {'Node Type': 'Other', 'Node Type ID': '42', 'Total Cost': 104619.210167, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Other', 'Node Type ID': '45', 'Total Cost': 104619.185167, 'Plan Rows': 2.0, 'Plans': [{'Node Type': 'Other', 'Node Type ID': '42', 'Total Cost': 103618.985167, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Hash Join', 'Node Type ID': '38', 'Total Cost': 103618.735167, 'Plan Rows': 32.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'title', 'Total Cost': 46533.15, 'Plan Rows': 1053515.0}, {'Node Type': 'Other', 'Node Type ID': '47', 'Total Cost': 53134.373917, 'Plan Rows': 32.0, 'Plans': [{'Node Type': 'Hash Join', 'Node Type ID': '38', 'Total Cost': 53134.373917, 'Plan Rows': 32.0, 'Plans': [{'Node Type': 'Hash Join', 'Node Type ID': '38', 'Total Cost': 15253.600333, 'Plan Rows': 5089.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'movie_info_idx', 'Total Cost': 13685.145833, 'Plan Rows': 575015.0}, {'Node Type': 'Other', 'Node Type ID': '47', 'Total Cost': 2.4125, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'info_type', 'Total Cost': 2.4125, 'Plan Rows': 1.0}]}]}, {'Node Type': 'Other', 'Node Type ID': '47', 'Total Cost': 37844.849833, 'Plan Rows': 1336.0, 'Plans': [{'Node Type': 'Hash Join', 'Node Type ID': '38', 'Total Cost': 37844.849833, 'Plan Rows': 1336.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'movie_companies', 'Total Cost': 37814.898958, 'Plan Rows': 5343.0}, {'Node Type': 'Other', 'Node Type ID': '47', 'Total Cost': 1.05, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'company_type', 'Total Cost': 1.05, 'Plan Rows': 1.0}]}]}]}]}]}]}]}]}]}}, {'Plan': {'Node Type': 'Other', 'Node Type ID': '42', 'Total Cost': 39801.111824, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Other', 'Node Type ID': '45', 'Total Cost': 39801.086824, 'Plan Rows': 2.0, 'Plans': [{'Node Type': 'Other', 'Node Type ID': '42', 'Total Cost': 38800.886824, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Nested Loop', 'Node Type ID': '36', 'Total Cost': 38800.636824, 'Plan Rows': 32.0, 'Plans': [{'Node Type': 'Hash Join', 'Node Type ID': '38', 'Total Cost': 38657.185594, 'Plan Rows': 32.0, 'Plans': [{'Node Type': 'Nested Loop', 'Node Type ID': '36', 'Total Cost': 38655.412094, 'Plan Rows': 131.0, 'Plans': [{'Node Type': 'Hash Join', 'Node Type ID': '38', 'Total Cost': 15253.600333, 'Plan Rows': 5089.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'movie_info_idx', 'Total Cost': 13685.145833, 'Plan Rows': 575015.0}, {'Node Type': 'Other', 'Node Type ID': '47', 'Total Cost': 2.4125, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'info_type', 'Total Cost': 2.4125, 'Plan Rows': 1.0}]}]}, {'Node Type': 'Other', 'Node Type ID': '24', 'Relation Name': 'movie_companies', 'Total Cost': 4.588509, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Bitmap Index Scan', 'Node Type ID': '23', 'Relation Name': 'movie_companies', 'Total Cost': 0.488259, 'Plan Rows': 5.0}]}]}, {'Node Type': 'Other', 'Node Type ID': '47', 'Total Cost': 1.05, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'company_type', 'Total Cost': 1.05, 'Plan Rows': 1.0}]}]}, {'Node Type': 'Other', 'Node Type ID': '24', 'Relation Name': 'title', 'Total Cost': 4.470351, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Bitmap Index Scan', 'Node Type ID': '23', 'Relation Name': 'title', 'Total Cost': 0.457601, 'Plan Rows': 1.0}]}]}]}]}]}}, {'Plan': {'Node Type': 'Other', 'Node Type ID': '42', 'Total Cost': 104619.210167, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Other', 'Node Type ID': '45', 'Total Cost': 104619.185167, 'Plan Rows': 2.0, 'Plans': [{'Node Type': 'Other', 'Node Type ID': '42', 'Total Cost': 103618.985167, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Hash Join', 'Node Type ID': '38', 'Total Cost': 103618.735167, 'Plan Rows': 32.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'title', 'Total Cost': 46533.15, 'Plan Rows': 1053515.0}, {'Node Type': 'Other', 'Node Type ID': '47', 'Total Cost': 53134.373917, 'Plan Rows': 32.0, 'Plans': [{'Node Type': 'Hash Join', 'Node Type ID': '38', 'Total Cost': 53134.373917, 'Plan Rows': 32.0, 'Plans': [{'Node Type': 'Hash Join', 'Node Type ID': '38', 'Total Cost': 15253.600333, 'Plan Rows': 5089.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'movie_info_idx', 'Total Cost': 13685.145833, 'Plan Rows': 575015.0}, {'Node Type': 'Other', 'Node Type ID': '47', 'Total Cost': 2.4125, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'info_type', 'Total Cost': 2.4125, 'Plan Rows': 1.0}]}]}, {'Node Type': 'Other', 'Node Type ID': '47', 'Total Cost': 37844.849833, 'Plan Rows': 1336.0, 'Plans': [{'Node Type': 'Hash Join', 'Node Type ID': '38', 'Total Cost': 37844.849833, 'Plan Rows': 1336.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'movie_companies', 'Total Cost': 37814.898958, 'Plan Rows': 5343.0}, {'Node Type': 'Other', 'Node Type ID': '47', 'Total Cost': 1.05, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'company_type', 'Total Cost': 1.05, 'Plan Rows': 1.0}]}]}]}]}]}]}]}]}]}}, {'Plan': {'Node Type': 'Other', 'Node Type ID': '42', 'Total Cost': 19532.637021, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Other', 'Node Type ID': '45', 'Total Cost': 19532.612021, 'Plan Rows': 2.0, 'Plans': [{'Node Type': 'Other', 'Node Type ID': '42', 'Total Cost': 18532.412021, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Nested Loop', 'Node Type ID': '36', 'Total Cost': 18532.162021, 'Plan Rows': 32.0, 'Plans': [{'Node Type': 'Hash Join', 'Node Type ID': '38', 'Total Cost': 18513.45993, 'Plan Rows': 32.0, 'Plans': [{'Node Type': 'Nested Loop', 'Node Type ID': '36', 'Total Cost': 18511.68643, 'Plan Rows': 131.0, 'Plans': [{'Node Type': 'Hash Join', 'Node Type ID': '38', 'Total Cost': 15253.600333, 'Plan Rows': 5089.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'movie_info_idx', 'Total Cost': 13685.145833, 'Plan Rows': 575015.0}, {'Node Type': 'Other', 'Node Type ID': '47', 'Total Cost': 2.4125, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'info_type', 'Total Cost': 2.4125, 'Plan Rows': 1.0}]}]}, {'Node Type': 'Index Scan', 'Node Type ID': '21', 'Relation Name': 'movie_companies', 'Total Cost': 0.630221, 'Plan Rows': 1.0}]}, {'Node Type': 'Other', 'Node Type ID': '47', 'Total Cost': 1.05, 'Plan Rows': 1.0, 'Plans': [{'Node Type': 'Seq Scan', 'Node Type ID': '19', 'Relation Name': 'company_type', 'Total Cost': 1.05, 'Plan Rows': 1.0}]}]}, {'Node Type': 'Index Scan', 'Node Type ID': '21', 'Relation Name': 'title', 'Total Cost': 0.57194, 'Plan Rows': 1.0}]}]}]}]}}]
    buffers = {'company_type_pkey': 1, 'info_type_pkey': 1, 'movie_companies_pkey': 1, 'company_id_movie_companies': 1, 'company_type_id_movie_companies': 1, 'movie_id_movie_companies': 279, 'movie_info_idx_pkey': 1, 'info_type_id_movie_info_idx': 1, 'movie_id_movie_info_idx': 6, 'title_pkey': 116, 'kind_id_title': 1, 'movie_companies': 572, 'movie_info_idx': 7935, 'title': 107, 'company_type': 1, 'info_type': 1}
    features = featurize(plans, buffers)
    print(plans)