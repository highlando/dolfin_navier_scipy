    # check for previously computed velocities
    if useolddata and lin_vel_point is None and not stokes_flow:
        try:
            datastrdict.update(dict(time=trange[-1]))
            cdatstr = get_datastring(**datastrdict)

            norm_nwtnupd = (dou.load_npa(cdatstr + '__norm_nwtnupd')).flatten()
            try:
                if norm_nwtnupd[0] is None:
                    norm_nwtnupd = 1.
            except IndexError:
                norm_nwtnupd = 1.

            dou.load_npa(cdatstr + '__vel')

            print('found vel files')
            print('norm of last Nwtn update: {0}'.format(norm_nwtnupd))
            print('... loaded from ' + cdatstr)

            if norm_nwtnupd < vel_nwtn_tol and not return_dictofvelstrs:
                return
            elif norm_nwtnupd < vel_nwtn_tol or treat_nonl_explct:
                # looks like converged / or semi-expl
                # -- check if all values are there
                # t0:
                datastrdict.update(dict(time=trange[0]))
                cdatstr = get_datastring(**datastrdict)
                dictofvelstrs = {}
                _atdct(dictofvelstrs, trange[0], cdatstr + '__vel')
                if return_dictofpstrs:
                    dictofpstrs = {}

                for t in trange:
                    datastrdict.update(dict(time=t))
                    cdatstr = get_datastring(**datastrdict)
                    # test if the vels are there
                    v_old = dou.load_npa(cdatstr + '__vel')
                    # update the dict
                    _atdct(dictofvelstrs, t, cdatstr + '__vel')
                    if return_dictofpstrs:
                        try:
                            p_old = dou.load_npa(cdatstr + '__p')
                            _atdct(dictofpstrs, t, cdatstr + '__p')
                        except:
                            p_old = get_pfromv(v=v_old, **gpfvd)
                            dou.save_npa(p_old, fstring=cdatstr + '__p')
                            _atdct(dictofpstrs, t, cdatstr + '__p')

                if return_dictofpstrs:
                    return dictofvelstrs, dictofpstrs
                else:
                    return dictofvelstrs

            # comp_nonl_semexp = False

        except IOError:
            norm_nwtnupd = 2
            print('no old velocity data found')

