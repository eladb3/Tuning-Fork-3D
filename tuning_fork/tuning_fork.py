from .optimization import *
from .gen_stl import *
from .utils import *

def generate_tuning_fork():
    pass
import inspect
    
class TuningFork:
    def __init__(
        self,
        E=200e9,
        rho=7.85e3,
        L=0.14,
        a=0.0025,
        N=256,
        save_dir='exp',
        basis_init='grid',
        lambdas_init='uniform',
        epsilon_init='const',
        A_weight=0.0,
        sigmoid_temp=1e0,
        entropy_weight=0.0,
        repulsion_weight=0.0,
        repulsion_epsilon=0.001,
        weight_decay=0.001,
        boundary_weight=0.,
        frequency = midi2hz(60),
        shape_function='default_circle',
        shape_boundary_val=1,
        shape_eps=0.0001,
        max_iters = 500,
        base_lr=1e-5,
    ):
        if shape_function == 'default_circle':
            shape_function=[[[0, 0, 1.2 * a]], ['circle']]
        for k,v in locals().items():
            setattr(self, k,v)
        self.sdf = None

    def get_params(self):
        self.C = C = compute_C(self.frequency, self.L, self.E, self.rho)

        params = {'C': C, 'N': self.N,
                'A_weight': self.A_weight,
                'sigmoid_t': self.sigmoid_temp,
                'entropy_weight': self.entropy_weight,
                'repulsion_weight': self.repulsion_weight, 'rep_epsilon': self.repulsion_epsilon,
                'weight_decay': self.weight_decay, 'boundary_weight': self.boundary_weight
                }
        params.update(dict(
            a=self.a,
            L=self.L,
            E=self.E,
            rho=self.rho,
            frequency=self.frequency,
            midi=hz2midi(self.frequency),
            N=self.N,
        ))
        return params

    def get_save_dirs(self):
        params = self.get_params()
        model_str = '#'.join('#'.join(str(el) for el in elem) for elem in params.items() if elem[0] != 'C')

        model_dir = self.save_dir + '/' + model_str + '/sdf'
        image_dir = self.save_dir + '/' + model_str + '/images'
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        return model_dir, image_dir

    def init_sdf(self):
        if getattr(self, 'sdf', None) is not None:
            return
        if self.shape_function is not None:
            indicator_fn = shape_indicator_fn(*self.shape_function, boundary_val=self.shape_boundary_val, eps=self.shape_eps)
        else:
            indicator_fn = None


        self.sdf = SDF(
            1600,
            a=2 * self.a,
            indicator_fn=indicator_fn,
            basis_init=self.basis_init,
            shape_function=self.shape_function,
        ).to(DEVICE)

    def optimize(self):
        params = self.get_params()
        nice_print('params:', params)
        model_dir, image_dir = self.get_save_dirs()
        self.init_sdf()
        sdf = self.sdf
        C = self.C
        curr_lr = self.base_lr

        iters = self.max_iters
        optimizer = optim.Adam(sdf.parameters(), lr=curr_lr)
        total_loss = 0.
        lr_change = {100: 1e-6}
        best_C_dist = float('inf')
        no_improvement = 0
        L = self.L
        E = self.E
        rho = self.rho
        for i in range(iters):
            print('iter', i)
            if i in lr_change:
                print('reducing lr to', lr_change[i])
                set_lr(optimizer, lr_change[i])

            S = sdf.get_set()
            optimizer.zero_grad()
            _params = {k:v for k,v in params.items() if k in inspect.signature(sdf.compute_loss).parameters.keys()}
            loss, C_eval, C_continuous = sdf.compute_loss(**_params)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            S_disc = (S > 0).astype(np.uint8) * 255
            S = (((S - S.min()) / (S.max() - S.min())) * 256).astype(np.uint8)
            if abs(C_eval - C) < best_C_dist:
                best_C_dist = abs(C_eval - C)
                no_improvement = 0
                best_C_image = S_disc
                save_pth = model_dir + '/best.pt'.format(i)
                torch.save(sdf.state_dict(), save_pth)

                # img = pil_img.fromarray(S_disc)
                save_pth = image_dir + '/best.pt'.format(i)
                cv2.imwrite(save_pth.replace('.pt', '') + '_{}_f{}.png'.format(i, hz2midi(self.frequency)), S_disc)
                # img_continuous = pil_img.fromarray(S)
                cv2.imwrite(save_pth.replace('.pt', '') + '_smooth_{}.png'.format(i), S)

                f_target = compute_f(L, E, rho, C)
                f_model = compute_f(L, E, rho, C_eval)
                m_target = hz2midi(f_target)
                m_model = hz2midi(f_model)
                print('improved, estimated error in hz:', f_target, 'vs', f_model, 'in tones:', m_target, 'vs', m_model)
                if i > 1:
                    plt.clf()
                f, axarr = plt.subplots(1, 2)
                axarr[0].imshow(S, cmap='gray')
                axarr[1].imshow(S_disc, cmap='gray')
                plt.show()
                if abs(m_target - m_model) <= 1e-1:
                    print("Finished")
                    break

            else:
                print('else')
                improved = False
                no_improvement += 1
                if no_improvement >= 20:
                    # curr_lr *= 0.5
                    # if curr_lr < 1e-6:
                    #     break
                    # set_lr(optimizer, curr_lr)
                    print('reduced learning rate to', curr_lr)
                    no_improvement = 0
    def get_cross_section(self):
        assert self.sdf is not None, "Please run optimization"
        m, mask = self.sdf.get_set(N=self.N,cut_to_shape=True, ret_mask=True)
        if mask is not None:
            mask = mask.astype(np.float32)
        m = (m > 0).astype(np.float32)
        return m, mask

    def build_voxel(self, **kwargs):
        resize = kwargs.pop('resize', None)
        m, mask = self.get_cross_section()
        if resize is not None:
            m = cv2.resize(m, (resize, resize), cv2.INTER_NEAREST)
            if mask is not None:
                mask = cv2.resize(mask, (resize, resize), cv2.INTER_NEAREST)

        default_args = dict(
            thickness=5, 
            close_sides=False, 
            sides_thickness=15, 
            handle_length=None, 
            prong_shape=None,
        )
        default_args.update(kwargs)
        kwargs = default_args
        if 'prong_length' not in kwargs:
            kwargs['prong_length'] = int((self.L / (2*self.a)) * m.shape[0])
        print(kwargs)
        voxel = gen_voxel(
            m,
            **kwargs,
        )
        return voxel

    def to_stl(self, **kwargs):
        outpath=kwargs.pop("outpath", None)
        simplify_mesh = kwargs.pop("simplify_mesh", True)
        if outpath is None:
            outpath = os.path.join(self.save_dir, "tuning-fork.stl")
        voxel = self.build_voxel(**kwargs)
        np2stl(voxel, outpath, simplify_mesh=simplify_mesh)

    def plot(self, resize=32, stl_path = None, **kwargs):
        if stl_path is None:
            p = "./tmp.stl"
            self.to_stl(
                outpath=p,
                close_sides=False,
                resize=resize,
                **kwargs,
            )
        else:
            p = stl_path
        show_voxel(path=p)        


    
class SimpleTuningFork:
    def __init__(
        self,
        E=200e9,
        rho=7.85e3,
        L=0.14,
        a=0.0025,
        frequency = midi2hz(60),
        N=256,
    ):
        for k,v in locals().items():
            setattr(self, k,v)
        self.C = C = (((a*2) ** 2) / 12)
        self.frequency = compute_f(L, E, rho, C)
        assert abs(C - compute_C(self.frequency, self.L, self.E, self.rho)) < 1e-4
    def get_params(self):
        params = dict(
            C=self.C,
            a=self.a,
            L=self.L,
            E=self.E,
            rho=self.rho,
            frequency=self.frequency,
            midi=hz2midi(self.frequency),
            N=self.N,
            
        )
        return params

    def get_save_dirs(self):
        params = self.get_params()
        model_str = '#'.join('#'.join(str(el) for el in elem) for elem in params.items() if elem[0] != 'C')

        model_dir = self.save_dir + '/' + model_str + '/sdf'
        image_dir = self.save_dir + '/' + model_str + '/images'
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        return model_dir, image_dir

    def get_cross_section(self):
        m = np.ones((self.N, self.N), dtype=np.float32)
        return m, None

    def build_voxel(self, **kwargs):
        resize = kwargs.pop('resize', None)
        m, mask = self.get_cross_section()
        if resize is not None:
            m = cv2.resize(m, (resize, resize), cv2.INTER_NEAREST)
            if mask is not None:
                mask = cv2.resize(mask, (resize, resize), cv2.INTER_NEAREST)

        default_args = dict(
            thickness=5, 
            close_sides=False, 
            sides_thickness=15, 
            handle_length=None, 
            prong_shape=None,
        )
        default_args.update(kwargs)
        kwargs = default_args
        if 'prong_length' not in kwargs:
            kwargs['prong_length'] = int((self.L / (2*self.a)) * m.shape[0])
        print(kwargs)
        voxel = gen_voxel(
            m,
            **kwargs,
        )
        return voxel

    def to_stl(self, **kwargs):
        outpath=kwargs.pop("outpath", None)
        simplify_mesh = kwargs.pop("simplify_mesh", True)
        target_number_of_triangles = kwargs.pop("target_number_of_triangles", None)
        if outpath is None:
            outpath = os.path.join(self.save_dir, "tuning-fork.stl")
        voxel = self.build_voxel(**kwargs)
        np2stl(voxel, outpath, simplify_mesh=simplify_mesh, target_number_of_triangles=target_number_of_triangles)
    
    def plot(self, resize=32):
        p = "./tmp.stl"
        self.to_stl(
            outpath=p,
            close_sides=False,
            resize=resize,
        )
        show_voxel(path=p)        

