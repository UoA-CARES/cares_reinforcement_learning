class SqVae(Autoencoder):
    def __init__(
        self,
        ae_type,
        loss_fn,
        img_size,
        latent_dim,
        device,
        k,
        dim_z,
        size_dict,
        param_var_q,
        flg_arelbo,
        num_rb,
        temperature,
        log_param_q_init,
    ):
        super().__init__(ae_type, loss_fn, img_size, latent_dim, k)
        self.device = device
        self.flg_arelbo = flg_arelbo
        if not self.flg_arelbo:
            self.logvar_x = nn.Parameter(torch.tensor(np.log(0.1)))

        # Data space
        self.dim_x = 3 * img_size[1] * img_size[1]
        self.dim_z = dim_z

        # Encoder/decoder
        self.param_var_q = param_var_q
        self.var_q = not self.param_var_q == ParamVarQ.GAUSSIAN_1
        self.encoder = SQEncoder(k, self.dim_z, num_rb, True, self.var_q).to(device)
        self.decoder = SQDecoder(k, self.dim_z, num_rb, True).to(device)
        self.apply(weight_init_sqvae)

        # Codebook
        self.size_dict = size_dict
        self.dim_dict = self.dim_z
        self.codebook = nn.Parameter(torch.randn(self.size_dict, self.dim_dict))
        self.log_param_q_scalar = nn.Parameter(torch.tensor(log_param_q_init))
        self.quantizer = SQVectorQuantizer(
            self.size_dict, self.dim_dict, temperature, self.param_var_q, device
        )

    def lookup(self, ids: torch.Tensor):
        return F.embedding(ids, self.codebook).permute(0, 3, 1, 2)

    def reconstruct_from_codes(self, codes):
        return self.decoder(self.lookup(codes))

    def reconstruct_average(self, x, num_samples=10):
        b, c, h, w = x.shape
        result = torch.empty((num_samples, b, c, h, w)).to(self.device)

        for i in range(num_samples):
            result[i] = self.reconstruct(x)
        return result.mean(0)

    def reconstruct(self, x):
        z_from_encoder = self.encoder(x)
        param_q = (
            torch.tensor([0.0], device=self.device).exp()
            + self.log_param_q_scalar.exp()
        )
        z_quantized, _, _ = self.quantizer(
            z_from_encoder, param_q, self.codebook, False, True
        )
        z_quantized = torch.flatten(z_quantized, start_dim=1)
        return self.decoder(z_quantized)

    def forward(
        self,
        states,
        storer=None,
        flg_train=True,
        flg_quant_det=True,
        detach=False,
        is_agent=False,
        **kwargs
    ):
        states = states / 255
        # Encoding
        if self.param_var_q == ParamVarQ.GAUSSIAN_1:
            z_from_encoder = self.encoder(states, detach)
            log_var_q = torch.tensor([0.0], device=self.device)
        else:
            z_from_encoder, log_var = self.encoder(states, detach)
            if self.param_var_q == ParamVarQ.GAUSSIAN_2:
                log_var_q = log_var.mean(dim=(1, 2, 3), keepdim=True)
            elif self.param_var_q == ParamVarQ.GAUSSIAN_3:
                log_var_q = log_var.mean(dim=1, keepdim=True)
            elif self.param_var_q == ParamVarQ.GAUSSIAN_4:
                log_var_q = log_var
            else:
                raise Exception("Undefined param_var_q")
        self.param_q = log_var_q.exp() + self.log_param_q_scalar.exp()

        # Quantization
        z_quantized, loss_latent, perplexity = self.quantizer(
            z_from_encoder, self.param_q, self.codebook, flg_train, flg_quant_det
        )

        z_quantized = torch.flatten(z_quantized, start_dim=1)

        # Decoding
        x_reconst = self.decoder(z_quantized)

        # Loss
        mse, loss = self.loss_fn(
            states,
            x_reconst,
            None,
            True,
            storer,
            self.flg_arelbo,
            loss_latent=loss_latent,
            is_agent=is_agent,
        )
        if detach:
            z_quantized = z_quantized.detach()

        return {
            "loss": loss,
            "z_vector": z_quantized,
            "rec_obs": x_reconst,
            "perplexity": perplexity,
            "mse_loss": mse,
        }
