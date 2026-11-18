"""Apply bounded training-loop corrections; no model/data/hyperparameter search."""
from pathlib import Path

def fix_reference_trainer(p):
    s=Path(p).read_text()
    def replace(a,b):
        nonlocal s
        assert a in s,a
        s=s.replace(a,b)
    replace('import argparse','import argparse\nimport json\nfrom training_checks import StepCounter')
    replace('    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing','    accumulate = 1  # one optimizer step per valid batch')
    replace("    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay","    # Keep the previous effective weight_decay unchanged (0.0005).")
    replace('                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())','                assert accumulate == 1')
    replace('    scaler = amp.GradScaler(enabled=False)','    step_counter = StepCounter(optimizer)\n    assert rank == -1 and not opt.resume and not opt.notest and not opt.nosave\n    epoch_log = save_dir / "epoch_accounting.jsonl"')
    replace('        model.train()','        model.train()\n        steps_before = step_counter.total\n        epoch_batches = epoch_images = epoch_labels = validation_calls = 0')
    replace('            ni = i + nb * epoch', '            epoch_batches += 1\n            epoch_images += len(imgs)\n            epoch_labels += len(targets)\n            assert tuple(imgs.shape[1:]) == (3, imgsz, imgsz)\n            ni = i + nb * epoch')
    replace('            scaler.scale(loss).backward()','            assert torch.isfinite(loss).all(), "Non-finite training loss"\n            loss.backward()  # upstream batch-scaled loss; no accumulation division')
    old='''            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)'''
    replace(old,'''            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema:
                ema.update(model)''')
    block="""        if epoch - best_epoch >= opt.patience:
            logger.info('Early stopping: epoch %d, best epoch %d', epoch + 1, best_epoch + 1)
            break

"""
    assert s.count(block) in (1,2)
    s=s.replace(block,'')
    replace('        # Scheduler\n', '        actual_steps = step_counter.verify_epoch(steps_before, epoch_batches, epoch_images, nb, len(dataset))\n\n        # Scheduler\n')
    replace('                results, maps, times = test.test(data_dict,','                validation_calls += 1\n                results, maps, times = test.test(data_dict,')
    replace("            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]", "            assert validation_calls == 1\n            fi = float(results[3])  # select only val mAP50-95\n            improved = fi > best_fitness")
    replace('            if fi > best_fitness:', '            if improved:')
    replace('                if best_fitness == fi:\n                    torch.save(ckpt, best)', '                if improved:\n                    torch.save(ckpt, best)')
    replace("                        'updates': ema.updates,", "                        'updates': ema.updates,\n                        'optimizer_steps': step_counter.total,\n                        'selection_metric': 'val_mAP50_95',")
    end='''        # end epoch ----------------------------------------------------------------------------------------------------
    # end training'''
    replace(end,'''        assert ema.updates == step_counter.total
        with epoch_log.open('a') as log:
            log.write(json.dumps(dict(epoch=epoch+1, batches=epoch_batches, images=epoch_images,
                augmented_labels=epoch_labels, expected_updates=epoch_batches, actual_updates=actual_steps,
                cumulative_updates=step_counter.total, validation_calls=validation_calls,
                last_saves=1, best_saves=int(improved), val_mAP50=float(results[2]),
                val_mAP50_95=fi, best_epoch=best_epoch+1, learning_rates=lr)) + '\\n')
        if epoch - best_epoch >= opt.patience:
            logger.info('Early stopping: epoch %d, best epoch %d', epoch + 1, best_epoch + 1)
            break

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training''')
    replace("default=20)","default=40)")
    assert s.count('if epoch - best_epoch >= opt.patience:')==1
    assert s.count('optimizer.step()')==1 and 'scaler.step' not in s
    Path(p).write_text(s)

if __name__=='__main__':
    fix_reference_trainer(Path(__file__).resolve().parent/'training_src/train.py')
