from process_stream_file import process_stream_file
# Helper function to process a file and store results
def process_and_store(stream_file_path, all_results, best_results, header, lock):
    results, none_results, file_header = process_stream_file(stream_file_path)

    if file_header and not header:
        with lock:
            if not header:
                header.append(file_header)

    with lock:
        all_results.extend(results)
        all_results.extend(none_results)

        # Update best_results to keep only the lowest combined metric for each event number
        best_results_dict = {result[1]: result for result in best_results}
        for result in results:
            event_number = result[1]
            if event_number not in best_results_dict or result[2] < best_results_dict[event_number][2]:
                best_results_dict[event_number] = result

        best_results[:] = list(best_results_dict.values())