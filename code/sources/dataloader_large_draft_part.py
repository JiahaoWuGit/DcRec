
def getSparseGraph_aug(self, aug_type = None, edge_drop_rate = 0, node_drop_rate = 0):
    print('loading augmentation adjacency matrix')
    try:
        preAdjMat = sp.load_npz(self.path + '/pref_s_pre_adj_mat_'+str(aug_type)+'_edRate_'+str(edge_drop_rate)+'_ndRate_'+str(edge_drop_rate)+'.npz')
        print("successfully loaded...")
        normAdj = preAdjMat
    except:
        print("generating augmentation adjacency matrix")
        s = time()
        adjMat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
        adjMat = adjMat.tolil()

        user_dim = self.trainUser
        item_dim = self.trainItem
        # ================Augmentation=================
        if aug_type == 'edge_drop':
            user_dim, item_dim = edgeDrop(user_dim, item_dim, edge_drop_rate)
        elif aug_type == 'node_drop':
            user_dim, item_dim = nodeDrop(self.n_users, self.m_items, user_dim, item_dim, node_drop_rate)
        elif aug_type == 'random_walk':
            user_dim, item_dim = nodeDrop(self.n_users, self.m_items, user_dim, item_dim, node_drop_rate)
            user_dim, item_dim = edgeDrop(user_dim, item_dim, edge_drop_rate)
        # ================Augmentation=================
        Aug_UserItemNet = csr_matrix((np.ones(len(user_dim)), (user_dim, item_dim)),
                                      shape=(self.n_users, self.m_items))

        R = Aug_UserItemNet.tolil()
        adjMat[:self.n_users, self.n_users:] = R
        adjMat[self.n_users:, :self.n_users] = R.T
        adjMat = adjMat.todok()

        rowsum = np.array(adjMat.sum(axis=1))
        dInv = np.power(rowsum, -0.5).flatten()
        dInv[np.isinf(dInv)] = 0.
        dMat = sp.diags(dInv)

        normAdj = dMat.dot(adjMat)
        normAdj = normAdj.dot(dMat)
        normAdj = normAdj.tocsr()
        end = time()
        print(f"costing {end - s}s, saved norm_mat...")
        sp.save_npz(self.path + '/pref_s_pre_adj_mat_'+str(aug_type)+'_edRate_'+str(edge_drop_rate)+'_ndRate_'+str(edge_drop_rate)+'.npz', normAdj)
    self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
    self.Graph = self.Graph.coalesce().to(world.device)
    return self.Graph


def getSparseGraph_aug_social(self, aug_type=None, edge_drop_rate=0, node_drop_rate=0):
    print('loading augmentation social adjacency matrix')
    try:
        preAdjMat = sp.load_npz(
            self.path + '/social_s_pre_adj_mat_' + str(aug_type) + '_edRate_' + str(edge_drop_rate) + '_ndRate_' + str(
                edge_drop_rate) + '.npz')
        print("successfully loaded...")
        normAdj = preAdjMat
    except:
        print("generating augmentation social adjacency matrix")
        s = time()
        adjMat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
        adjMat = adjMat.tolil()

        trustor = self.trustor
        trustee = self.trustee
        # ================Augmentation=================
        if aug_type == 'edge_drop':
            trustor, trustee = edgeDrop(trustor, trustee, edge_drop_rate)
        elif aug_type == 'node_drop':
            trustor, trustee = nodeDrop_social(self.n_users, trustor, trustee, node_drop_rate)
        elif aug_type == 'random_walk':
            trustor, trustee = nodeDrop_social(self.n_users, trustor, trustee, node_drop_rate)
            trustor, trustee = edgeDrop(trustor, trustee, edge_drop_rate)
        # ================Augmentation=================

        Aug_SocialNet = csr_matrix((np.ones(len(self.trustNet)), (trustor, trustee)),
                                        shape=(self.n_users, self.n_users))

        R = Aug_SocialNet.tolil()
        adjMat[:self.n_users, self.n_users:] = R
        adjMat[self.n_users:, :self.n_users] = R.T
        adjMat = adjMat.todok()

        rowsum = np.array(adjMat.sum(axis=1))
        dInv = np.power(rowsum, -0.5).flatten()
        dInv[np.isinf(dInv)] = 0.
        dMat = sp.diags(dInv)

        normAdj = dMat.dot(adjMat)
        normAdj = normAdj.dot(dMat)
        normAdj = normAdj.tocsr()
        end = time()
        print(f"costing {end - s}s, saved norm_mat...")
        sp.save_npz(
            self.path + '/social_s_pre_adj_mat_' + str(aug_type) + '_edRate_' + str(edge_drop_rate) + '_ndRate_' + str(
                edge_drop_rate) + '.npz', normAdj)
    self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
    self.Graph = self.Graph.coalesce().to(world.device)
    return self.Graph


