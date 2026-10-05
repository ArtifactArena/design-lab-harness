"""Named physical state and interaction observations, shared by native and WASM."""
from copy import deepcopy
import numpy as np
import mujoco
from mjarena.envs.surface_distance import SurfaceQueries

SURFACE_CUTOFF = 2.0
SURFACE_LIMIT = 32


def _name(model, kind, index, prefix=''):
    name = mujoco.mj_id2name(model, kind, int(index))
    if name:
        return name[len(prefix):] if prefix and name.startswith(prefix) else name
    # Author-supplied names can use our generated-name syntax. Never overwrite
    # another record (or redirect a body/joint/site reference) on that account.
    name = f'@{int(kind)}:{index}'
    while (mujoco.mj_name2id(model, kind, name) >= 0
           or (prefix and mujoco.mj_name2id(model, kind, prefix + name) >= 0)):
        name += '~'
    return name


def _quaternion(matrix):
    # WXYZ, stable near pi rotations; result and its negation represent the same pose.
    m = np.asarray(matrix).reshape(3, 3)
    k = np.array([[m[0,0]-m[1,1]-m[2,2],m[0,1]+m[1,0],m[0,2]+m[2,0],m[2,1]-m[1,2]],
                  [m[0,1]+m[1,0],m[1,1]-m[0,0]-m[2,2],m[1,2]+m[2,1],m[0,2]-m[2,0]],
                  [m[0,2]+m[2,0],m[1,2]+m[2,1],m[2,2]-m[0,0]-m[1,1],m[1,0]-m[0,1]],
                  [m[2,1]-m[1,2],m[0,2]-m[2,0],m[1,0]-m[0,1],np.trace(m)]])
    _, v = np.linalg.eigh(k)
    q = v[:, -1][[3,0,1,2]]
    return q if q[0] >= 0 else -q


class DetailedObservations:
    def __init__(self, env):
        self.env = env; self.model = env.model
        self.data = mujoco.MjData(self.model)
        self.queries = SurfaceQueries(self.model)
        self.prefixes = [env.red_contender.prefix, env.blue_contender.prefix]
        self.owners = {}
        self.ids = {}
        self.static = {}
        self.cached = None
        self.impulses = {}
        self.latest_contacts = []
        self.interval_seconds = 0.
        self.mass = np.asarray(self.model.body_mass).copy()
        self.geom_body = np.asarray(self.model.geom_bodyid).copy()
        self.radii = np.asarray(self.model.geom_rbound).copy()
        for prefix in self.prefixes:
            body_ids = [i for i in range(1, self.model.nbody) if env._body_belongs_to_contender(i, prefix)]
            geom_ids = list(env._contender_geom_ids[prefix])
            self.ids[prefix] = (body_ids, geom_ids)
            self.owners.update({g: prefix for g in geom_ids})
            self.static[prefix] = self._definitions(prefix, body_ids, geom_ids)

    def close(self):
        close = getattr(self.data, 'close', None)
        if close:
            close()

    @property
    def control_dt(self):
        """Simulated duration of one controller interval, matching env.step."""
        physics_dt=float(self.model.opt.timestep)
        substeps=1 if physics_dt<=0 else max(1,int(round(self.env.control_timestep/physics_dt)))
        return physics_dt*substeps*self.env.apply_n_repeated_actions

    def _definitions(self, prefix, bodies, geoms):
        m = self.model
        bn = lambda i: _name(m, mujoco.mjtObj.mjOBJ_BODY, i, prefix)
        jn = lambda i: _name(m, mujoco.mjtObj.mjOBJ_JOINT, i, prefix)
        sn = lambda i: _name(m, mujoco.mjtObj.mjOBJ_SITE, i, prefix)
        gn = lambda i: _name(m, mujoco.mjtObj.mjOBJ_GEOM, i, prefix)
        out = {'bodies': {}, 'geoms': {}, 'joints': {}, 'tendons': {}, 'motors': {}, 'sites': {}}
        for i in bodies:
            parent = int(m.body_parentid[i])
            out['bodies'][bn(i)] = {'id': i, 'parent': bn(parent) if parent in bodies else None,
                'mass': float(m.body_mass[i]), 'local_com': np.array(m.body_ipos[i]),
                'inertia_diagonal': np.array(m.body_inertia[i]), 'inertia_quaternion': np.array(m.body_iquat[i])}
        kinds = {int(getattr(mujoco.mjtGeom, 'mjGEOM_'+k.upper())): k for k in ('plane','sphere','capsule','ellipsoid','cylinder','box','mesh','sdf')}
        for g in geoms:
            record = {'id': g, 'body': bn(int(m.geom_bodyid[g])), 'type': kinds[int(m.geom_type[g])],
                'size': np.array(m.geom_size[g]), 'local_position': np.array(m.geom_pos[g]),
                'local_quaternion': np.array(m.geom_quat[g]), 'friction': np.array(m.geom_friction[g])}
            mid = int(m.geom_dataid[g])
            if g in self.queries.meshes:
                va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                record.update(vertices=np.asarray(m.mesh_vert)[va:va+vn].copy(), faces=np.asarray(m.mesh_face)[fa:fa+fn].copy())
            out['geoms'][gn(g)] = record
        joint_types = {0:'free',1:'ball',2:'slide',3:'hinge'}
        for j in range(m.njnt):
            if int(m.jnt_bodyid[j]) not in bodies or int(m.jnt_type[j]) == 0:
                continue
            qadr, dadr = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            ball = int(m.jnt_type[j]) == 1
            out['joints'][jn(j)] = {'id':j,'body':bn(int(m.jnt_bodyid[j])), 'type':joint_types[int(m.jnt_type[j])],
                'axis':np.array(m.jnt_axis[j]), 'anchor':np.array(m.jnt_pos[j]),
                'limited':bool(m.jnt_limited[j]), 'range':np.array(m.jnt_range[j]),
                'reference':np.asarray(m.qpos0)[qadr:qadr+(4 if ball else 1)].copy(),
                'stiffness':float(m.jnt_stiffness[j]),
                'damping':np.asarray(m.dof_damping)[dadr:dadr+(3 if ball else 1)].copy(),
                'frictionloss':np.asarray(m.dof_frictionloss)[dadr:dadr+(3 if ball else 1)].copy()}
        for i in range(m.nsite):
            if int(m.site_bodyid[i]) in bodies:
                out['sites'][sn(i)] = {'id':i,'body':bn(int(m.site_bodyid[i])), 'local_position':np.array(m.site_pos[i]), 'local_quaternion':np.array(m.site_quat[i])}
        wrap_names = {0:'none',1:'joint',2:'pulley',3:'site',4:'sphere',5:'cylinder'}
        tendon_ids = set()
        for t in range(m.ntendon):
            wraps = range(int(m.tendon_adr[t]), int(m.tendon_adr[t])+int(m.tendon_num[t]))
            owners = []
            for w in wraps:
                kind, obj = int(m.wrap_type[w]), int(m.wrap_objid[w])
                if kind == 1: owners.append(int(m.jnt_bodyid[obj]))
                elif kind == 3: owners.append(int(m.site_bodyid[obj]))
                elif kind in (4,5): owners.append(int(m.geom_bodyid[obj]))
            if not owners or not all(body in bodies for body in owners):
                continue
            tendon_ids.add(t)
            path = []
            for w in range(int(m.tendon_adr[t]), int(m.tendon_adr[t])+int(m.tendon_num[t])):
                kind, obj, prm = int(m.wrap_type[w]),int(m.wrap_objid[w]),float(m.wrap_prm[w])
                item = {'type':wrap_names[kind]}
                if kind == 1:item.update(joint=jn(obj),coef=prm)
                elif kind == 2:item['divisor']=prm
                elif kind == 3:item['site']=sn(obj)
                elif kind in (4,5):item.update(geom=gn(obj),sidesite=sn(int(prm)) if prm >= 0 else None)
                path.append(item)
            out['tendons'][_name(m,mujoco.mjtObj.mjOBJ_TENDON,t,prefix)] = {'id':t,'type':'fixed' if all(p['type']=='joint' for p in path) else 'spatial',
                'path':path,'stiffness':float(m.tendon_stiffness[t]),'damping':float(m.tendon_damping[t]),
                'frictionloss':float(m.tendon_frictionloss[t]),'springlength':np.array(m.tendon_lengthspring[t]),
                'limited':bool(m.tendon_limited[t]),'range':np.array(m.tendon_range[t])}
        for a in range(m.nu):
            transmission = int(m.actuator_trntype[a]); obj = int(m.actuator_trnid[a,0])
            transmissions = {
                int(mujoco.mjtTrn.mjTRN_JOINT): ('joint', mujoco.mjtObj.mjOBJ_JOINT),
                int(mujoco.mjtTrn.mjTRN_JOINTINPARENT): ('jointinparent', mujoco.mjtObj.mjOBJ_JOINT),
                int(mujoco.mjtTrn.mjTRN_TENDON): ('tendon', mujoco.mjtObj.mjOBJ_TENDON),
                int(mujoco.mjtTrn.mjTRN_SITE): ('site', mujoco.mjtObj.mjOBJ_SITE),
                int(mujoco.mjtTrn.mjTRN_SLIDERCRANK): ('slidercrank', mujoco.mjtObj.mjOBJ_SITE),
            }
            kind, target_type = transmissions[transmission]
            if kind in ('joint', 'jointinparent'):
                belongs = int(m.jnt_bodyid[obj]) in bodies
            elif kind == 'tendon':
                belongs = obj in tendon_ids
            else:
                belongs = int(m.site_bodyid[obj]) in bodies
            if not belongs: continue
            record = {'id':a, 'transmission':kind, 'target':_name(m,target_type,obj,prefix),
                'gear':np.array(m.actuator_gear[a]), 'ctrl_limited':bool(m.actuator_ctrllimited[a]),
                'ctrl_range':np.array(m.actuator_ctrlrange[a])}
            if kind in ('site', 'slidercrank'):
                record['refsite' if kind == 'site' else 'slidersite'] = sn(int(m.actuator_trnid[a,1]))
            if kind == 'slidercrank':
                record['cranklength'] = float(m.actuator_cranklength[a])
            out['motors'][_name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,a,prefix)] = record
        return out

    def begin_interval(self):
        self.impulses = {}; self.interval_seconds = 0.; self.cached = None

    def capture_contacts(self, integrate=False):
        """Force on geom2, world frame. Preserve every current contact separately."""
        m, d = self.model, self.env.data
        records=[]; buf=np.zeros(6); dt=float(m.opt.timestep)
        for i in range(int(d.ncon)):
            c=d.contact[i]; a,b=int(c.geom1),int(c.geom2)
            if a not in self.owners and b not in self.owners:continue
            if int(c.efc_address) < 0:continue
            mujoco.mj_contactForce(m,d,i,buf)
            frame=np.asarray(c.frame).reshape(3,3)
            force=frame.T @ buf[:3]; torque=frame.T @ buf[3:]
            pos=np.array(c.pos); normal=frame[0].copy()
            records.append((a,b,pos,normal,force.copy(),torque.copy(),float(c.dist)))
            if integrate:
                item=self.impulses.setdefault((a,b), {'impulse':np.zeros(3),'torque_impulse':np.zeros(3),'position_sum':np.zeros(3),'weight':0.,'duration':0.})
                item['impulse']+=force*dt; item['torque_impulse']+=torque*dt
                weight=float(np.linalg.norm(force))*dt
                item['position_sum']+=pos*weight;item['weight']+=weight;item['duration']+=dt
        self.latest_contacts=records
        if integrate:self.interval_seconds+=dt

    def _other(self, gid, prefix):
        owner=self.owners.get(gid)
        if owner == prefix: kind='self'
        elif owner:kind='opponent'
        elif gid == self.env._boundary_geom_id:kind='platform'
        elif gid == self.env.outside_floor_gid:kind='floor'
        else:kind='environment'
        return kind, _name(self.model,mujoco.mjtObj.mjOBJ_GEOM,gid,owner or '')

    def contacts_for(self,prefix):
        result=[]
        for a,b,pos,normal,force,torque,distance in self.latest_contacts:
            for mine,other,sign in ((a,b,-1),(b,a,1)):
                if self.owners.get(mine)!=prefix:continue
                kind,name=self._other(other,prefix)
                result.append({'my_geom':_name(self.model,mujoco.mjtObj.mjOBJ_GEOM,mine,prefix), 'other_kind':kind,'other_geom':name,
                    'position':pos.copy(),'normal_toward_me':sign*normal,'force_on_me':sign*force,'torque_on_me':sign*torque,'signed_distance':distance})
        return result

    def impulses_for(self,prefix):
        result=[]
        for (a,b),r in sorted(self.impulses.items()):
            for mine,other,sign in ((a,b,-1),(b,a,1)):
                if self.owners.get(mine)!=prefix:continue
                kind,name=self._other(other,prefix)
                result.append({'my_geom':_name(self.model,mujoco.mjtObj.mjOBJ_GEOM,mine,prefix),'other_kind':kind,'other_geom':name,
                    'impulse_on_me':sign*r['impulse'],'torque_impulse_on_me':sign*r['torque_impulse'],
                    'position':r['position_sum']/r['weight'] if r['weight']>0 else None})
        return result

    def _robot(self,prefix):
        m,d=self.model,self.data
        out=deepcopy(self.static[prefix]); bids,gids=self.ids[prefix]
        mass=sum(self.mass[b] for b in bids); com=np.zeros(3); velocity=np.zeros(3)
        for name,r in out['bodies'].items():
            b=r.pop('id'); origin=np.array(d.xpos[b]); center=np.array(d.xipos[b])
            v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_XBODY,b,v,0)
            cv=v[3:]+np.cross(v[:3],center-origin)
            r.update(position=origin,quaternion=np.array(d.xquat[b]),linear_velocity=v[3:].copy(),angular_velocity=v[:3].copy(),com_position=center,com_velocity=cv)
            com+=self.mass[b]*center;velocity+=self.mass[b]*cv
        for name,r in out['geoms'].items():
            g=r.pop('id');v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_GEOM,g,v,0)
            r.update(position=np.array(d.geom_xpos[g]),quaternion=_quaternion(d.geom_xmat[g]),linear_velocity=v[3:].copy(),angular_velocity=v[:3].copy())
        for name,r in out['joints'].items():
            j=r.pop('id');qa=int(m.jnt_qposadr[j]);da=int(m.jnt_dofadr[j]);ball=r['type']=='ball'
            r['position']=np.asarray(d.qpos)[qa:qa+4].copy() if ball else float(d.qpos[qa])
            r['velocity']=np.asarray(d.qvel)[da:da+3].copy() if ball else float(d.qvel[da])
        for r in out['tendons'].values():
            t=r.pop('id');r.update(length=float(d.ten_length[t]),velocity=float(d.ten_velocity[t]))
        for r in out['motors'].values():r.pop('id')
        for r in out['sites'].values():
            i=r.pop('id');v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_SITE,i,v,0)
            r.update(position=np.array(d.site_xpos[i]),quaternion=_quaternion(d.site_xmat[i]),
                linear_velocity=v[3:].copy(),angular_velocity=v[:3].copy())
        root=self.env._root_body_ids[prefix];root_name=_name(m,mujoco.mjtObj.mjOBJ_BODY,root,prefix)
        out.update(mass=float(mass),com_position=com/mass,com_velocity=velocity/mass,root_body=root_name)
        return out

    def _point_velocity(self,gid,point,robot):
        name=_name(self.model,mujoco.mjtObj.mjOBJ_GEOM,gid,self.owners[gid]);g=robot['geoms'][name]
        return g['linear_velocity']+np.cross(g['angular_velocity'],point-g['position'])

    def snapshot(self):
        if self.cached is not None:return self.cached
        if self.env.t == 0:
            self.capture_contacts()
        m,d=self.model,self.data
        d.qpos[:]=np.asarray(self.env.data.qpos);d.qvel[:]=np.asarray(self.env.data.qvel)
        if m.nmocap:
            d.mocap_pos[:]=np.asarray(self.env.data.mocap_pos);d.mocap_quat[:]=np.asarray(self.env.data.mocap_quat)
        # Separate data keeps observation queries from changing the match solver.
        mujoco.mj_forward(m,d)
        robots={p:self._robot(p) for p in self.prefixes}
        self.queries.update(d)
        left,right=self.prefixes;pairs=[];minimum=float('inf')
        ra,rb=self.ids[left][1],self.ids[right][1]
        # Broad-phase lower bounds prune exact geometry queries while preserving the global minimum.
        centers=self.queries.positions
        total=0
        for a in ra:
            delta=centers[rb]-centers[a]
            lower=np.maximum(np.linalg.norm(delta,axis=1)-self.radii[a]-self.radii[rb],0)
            for index in np.argsort(lower, kind='stable'):
                if lower[index] > max(minimum,SURFACE_CUTOFF):break
                b=rb[index]
                distance,pa,pb=self.queries.distance(d,a,b)
                if distance < minimum:minimum=distance
                if distance <= SURFACE_CUTOFF:
                    direction=(pb-pa)/np.linalg.norm(pb-pa) if distance>1e-8 else np.zeros(3)
                    closing=float(direction @ (self._point_velocity(a,pa,robots[left])-self._point_velocity(b,pb,robots[right])))
                    pairs.append((distance,a,b,pa,pb,closing));total+=1
                    if len(pairs) > 2*SURFACE_LIMIT:
                        pairs.sort(key=lambda p:(p[0],p[1],p[2]));pairs=pairs[:SURFACE_LIMIT]
        pairs.sort(key=lambda p:(p[0],p[1],p[2]))
        self.cached=(robots,pairs[:SURFACE_LIMIT],minimum,total)
        return self.cached

    def for_robot(self,prefix):
        robots,pairs,minimum,total=self.snapshot();left,right=self.prefixes
        other=right if prefix==left else left
        listed=[]
        for distance,a,b,pa,pb,closing in pairs:
            if prefix!=left:a,b,pa,pb=b,a,pb,pa
            listed.append({'my_geom':_name(self.model,mujoco.mjtObj.mjOBJ_GEOM,a,prefix),
                'opponent_geom':_name(self.model,mujoco.mjtObj.mjOBJ_GEOM,b,other),
                'distance':distance,'my_point':pa.copy(),'opponent_point':pb.copy(),'closing_speed':closing})
        env=self.env
        # Report simulated seconds, including integer substep rounding and
        # repeated actions, rather than the requested nominal control period.
        dt=self.control_dt
        platform_center=np.array(env.data.geom_xpos[env._boundary_geom_id])
        platform_center[2]=env.ring_top_z
        platform={'shape':env._boundary_shape,'center':platform_center,
            'radius':env.ring_radius,'half_extents':np.array(env.ring_half_extents),'top_z':env.ring_top_z,
            'floor_z':float(env.data.geom_xpos[env.outside_floor_gid,2]) if env.outside_floor_gid>=0 else None}
        return {'control_dt':dt,'elapsed_time':env.t*dt,'time_remaining':max(0,env.max_steps-env.t)*dt,
            'platform':platform,'my_robot':deepcopy(robots[prefix]),'opponent_robot':deepcopy(robots[other]),
            'my_mass':robots[prefix]['mass'],'opponent_mass':robots[other]['mass'],
            'my_com_velocity':robots[prefix]['com_velocity'].copy(),'opponent_com_velocity':robots[other]['com_velocity'].copy(),
            'opponent_surface_distance':minimum,'opponent_proximity':listed,
            'proximity_cutoff':SURFACE_CUTOFF,'proximity_limit':SURFACE_LIMIT,'proximity_truncated':total>SURFACE_LIMIT,
            'contacts':self.contacts_for(prefix),'contact_impulses':self.impulses_for(prefix),
            'contact_interval':self.interval_seconds}
